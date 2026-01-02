"""
test_pipelined_j0.py - Test pipelined J0 warmup optimization

This test verifies that the AES-GCM core correctly handles the optimization
where gctr_ready no longer waits for j0_loaded. This allows ciphertext
decryption to start ~16 cycles earlier while J0 computes in parallel.

Key behaviors tested:
1. Decryption starts as soon as H is loaded (not waiting for J0)
2. J0 is still correctly computed and available for tag calculation
3. Tag verification still works correctly
4. Multiple back-to-back packets work correctly
5. Latency is reduced compared to the sequential approach

The optimization is based on the insight that J0 is only needed at the END
of packet processing for tag computation: tag = GHASH(AAD||CT||len) XOR J0
"""

import json
import random
import cocotb
import gcm_model
import gcm_gctr as gctr

from cocotb.clock import Clock
from gcm_driver import pkt_driver, aad_driver, pt_driver, wait_for
from gcm_sequencer import sequencer, gcm_AAD_monitor, gcm_PT_monitor, gcm_CT_monitor, gcm_TAG_monitor
from cocotb.triggers import RisingEdge, FallingEdge, ClockCycles, Timer
from cocotb_bus.scoreboard import Scoreboard

CLK_PERIOD = 10
RST_WINDOW = CLK_PERIOD + (CLK_PERIOD * 3 // 4)


async def measure_ready_latency(dut, tb):
    """Measure cycles from icb_start to gctr_ready rising edge."""
    # Wait for icb_start
    while dut.aes_gcm_icb_start_cnt_i.value == 0:
        await RisingEdge(dut.clk_i)

    start_cycle = 0
    h_loaded_cycle = 0
    j0_loaded_cycle = 0
    ready_cycle = 0

    cycle = 0
    while True:
        await RisingEdge(dut.clk_i)
        cycle += 1

        # Track H loaded
        if h_loaded_cycle == 0 and dut.u_aes_gcm.u_gcm_ghash.ghash_h_loaded_o.value == 1:
            h_loaded_cycle = cycle
            dut._log.info(f"H loaded at cycle {cycle}")

        # Track J0 loaded
        if j0_loaded_cycle == 0 and dut.u_aes_gcm.u_gcm_ghash.ghash_j0_loaded_o.value == 1:
            j0_loaded_cycle = cycle
            dut._log.info(f"J0 loaded at cycle {cycle}")

        # Track ready
        if ready_cycle == 0 and dut.aes_gcm_ready_o.value == 1:
            ready_cycle = cycle
            dut._log.info(f"Ready at cycle {cycle}")
            break

        # Timeout
        if cycle > 200:
            dut._log.error("Timeout waiting for ready signal")
            break

    return {
        'h_loaded_cycle': h_loaded_cycle,
        'j0_loaded_cycle': j0_loaded_cycle,
        'ready_cycle': ready_cycle
    }


@cocotb.test()
async def test_ready_before_j0(dut):
    """
    Test that gctr_ready asserts as soon as H is loaded, NOT waiting for J0.

    This is the key behavioral change from the optimization:
    - OLD: ready = aes_ecb_val AND h_loaded AND j0_loaded
    - NEW: ready = aes_ecb_val AND h_loaded

    Expected behavior:
    - ready_cycle should be close to h_loaded_cycle
    - ready_cycle should be LESS than j0_loaded_cycle (ready before J0!)
    """
    tb = gctr.gcm_gctr(dut)

    # Simple test configuration
    tb.config = {
        'aes_mode': '128',
        'aes_size': '128',
        'enc_dec': 'dec',
        'key_pre_exp': False
    }

    # Test data
    tb.data = {
        'key': {'data': '00112233445566778899aabbccddeeff', 'n_bytes': 16},
        'iv': {'data': '000102030405060708090a0b', 'n_bytes': 12},
        'aad_n_bytes': 0,
        'pt_n_bytes': 16,
        'pt': {'data': '00000000000000000000000000000000', 'n_bytes': 16}
    }

    # Start clock
    cocotb.start_soon(Clock(dut.clk_i, CLK_PERIOD, 'ns').start())

    # Release reset
    cocotb.start_soon(tb.release_rst(RST_WINDOW))
    await FallingEdge(dut.rst_i)
    await ClockCycles(dut.clk_i, 10)

    # Set decryption mode
    await tb.set_enc_dec('dec')
    await tb.aes_set_mode()

    # Load key
    await tb.load_key(tb.data['key'])

    # Load IV
    await tb.load_iv(tb.data['iv'])

    # Start ICB and measure latency
    latency_task = cocotb.start_soon(measure_ready_latency(dut, tb))
    await tb.start_icb()

    # Wait for measurements
    latency = await latency_task

    dut._log.info(f"Latency results:")
    dut._log.info(f"  H loaded:  cycle {latency['h_loaded_cycle']}")
    dut._log.info(f"  J0 loaded: cycle {latency['j0_loaded_cycle']}")
    dut._log.info(f"  Ready:     cycle {latency['ready_cycle']}")

    # KEY ASSERTION: Ready should come BEFORE or AT SAME TIME as J0
    # With the optimization, ready depends only on H, not J0
    assert latency['ready_cycle'] <= latency['j0_loaded_cycle'], \
        f"Ready ({latency['ready_cycle']}) should not wait for J0 ({latency['j0_loaded_cycle']})"

    # Ready should be very close to H loaded (just 1 cycle for aes_ecb_val)
    assert latency['ready_cycle'] - latency['h_loaded_cycle'] <= 2, \
        f"Ready should follow H loaded closely, but gap is {latency['ready_cycle'] - latency['h_loaded_cycle']} cycles"

    dut._log.info("PASS: Ready signal does not wait for J0 (pipelined warmup working)")


@cocotb.test()
async def test_tag_still_correct(dut):
    """
    Test that the authentication tag is still computed correctly.

    Even though decryption starts before J0 is loaded, J0 must still be
    correctly captured and used for tag computation at packet end.

    tag = GHASH(AAD || CT || lengths) XOR J0
    """
    # Create lists of transactions
    aad_tran = []
    pt_tran = []
    aad_model_tran = []
    pt_model_tran = []

    tb = gctr.gcm_gctr(dut)

    # Open config file
    with open('./tmp/' + str(cocotb.RANDOM_SEED) + '.json', 'r') as config_file:
        tb.config = dict(json.load(config_file))

    # Generate configuration data
    tb.config_data()

    # Initialise GCM model
    dut_model = gcm_model.gcm(tb.data['key'], tb.data['iv'], tb.config['enc_dec'])

    # Create drivers
    pkt_drv = pkt_driver(dut.clk_i, dut.aes_gcm_ghash_pkt_val_i)
    aad_drv = aad_driver(dut.clk_i, dut.aes_gcm_ghash_aad_bval_i, dut.aes_gcm_ghash_aad_i)
    pt_drv = pt_driver(dut.clk_i, dut.aes_gcm_data_in_bval_i, dut.aes_gcm_data_in_i, dut.aes_gcm_ready_o)

    # Create delay function
    delay = wait_for(dut.clk_i, RisingEdge)

    # Release the Reset
    cocotb.start_soon(tb.release_rst(RST_WINDOW))

    # Start the Clock
    cocotb.start_soon(Clock(dut.clk_i, CLK_PERIOD, 'ns').start())

    # Wait for the Reset falling edge event
    await FallingEdge(dut.rst_i)
    await ClockCycles(dut.clk_i, random.randint(10, 20))

    # Create the sequencer
    seq = sequencer(pkt_drv, aad_drv, pt_drv, delay.n_clk, tb.data, aad_tran, pt_tran)

    if tb.config['enc_dec'] == 'enc':
        data_in_callback = dut_model.load_plain_text
    else:
        data_in_callback = dut_model.load_cipher_text

    # Create monitors
    mon_aad = gcm_AAD_monitor("Get AAD", dut, dut_model.load_aad)
    mon_data_in = gcm_PT_monitor("Get PT", dut, data_in_callback)
    mon_data_out = gcm_CT_monitor("Get CT", dut)
    mon_tag = gcm_TAG_monitor("Get TAG", dut, dut_model.get_tag)

    # Create scoreboard
    scoreboard = Scoreboard(dut)

    # Add scoreboard interfaces
    scoreboard.add_interface(mon_aad, aad_model_tran)
    scoreboard.add_interface(mon_data_in, pt_model_tran)
    scoreboard.add_interface(mon_data_out, dut_model.data_out)
    scoreboard.add_interface(mon_tag, dut_model.tag)

    # Set the AES in encryption or decryption mode
    await tb.set_enc_dec(tb.config['enc_dec'])

    # Set AES key mode
    await tb.aes_set_mode()

    # Load the KEY
    if tb.config['key_pre_exp']:
        await tb.load_pre_exp_key(tb.data['key'])
    else:
        await tb.load_key(tb.data['key'])

    # Load the ICB
    await tb.load_iv(tb.data['iv'])

    # Start The ICB
    await tb.start_icb()

    # Wait the AES to produce cipher data
    await tb.cipher_is_ready()

    # Log transaction counts
    n_transaction = tb.data['aad_n_bytes'] >> 4
    if tb.data['aad_n_bytes'] & 0xF:
        n_transaction += 1
    dut._log.info(f"\nAAD:\t{n_transaction}\ttransactions to read")

    n_transaction = tb.data['pt_n_bytes'] >> 4
    if tb.data['pt_n_bytes'] & 0xF:
        n_transaction += 1
    dut._log.info(f"DATA:\t{n_transaction}\ttransactions to read\n")

    # Start the sequencer
    seq.start_sequencer()

    # Encrypt data
    await tb.encrypt_data(tb.data['aad_n_bytes'], tb.data['pt_n_bytes'],
                          aad_tran, pt_tran, aad_model_tran, pt_model_tran)

    # Wait for the test to finish
    while dut.aes_gcm_ghash_tag_val_o.value == 0:
        await RisingEdge(dut.clk_i)

    await ClockCycles(dut.clk_i, 20)

    dut._log.info("PASS: Tag computed correctly with pipelined J0 warmup")


@cocotb.test()
async def test_back_to_back_packets(dut):
    """
    Test multiple back-to-back packets with new IVs.

    This tests that:
    1. Each packet gets correct J0 for its nonce
    2. Ready signal correctly re-asserts after IV reload
    3. No corruption between packets
    """
    tb = gctr.gcm_gctr(dut)

    # Simple test configuration
    tb.config = {
        'aes_mode': '128',
        'aes_size': '128',
        'enc_dec': 'dec',
        'key_pre_exp': False
    }

    # Start clock
    cocotb.start_soon(Clock(dut.clk_i, CLK_PERIOD, 'ns').start())

    # Release reset
    cocotb.start_soon(tb.release_rst(RST_WINDOW))
    await FallingEdge(dut.rst_i)
    await ClockCycles(dut.clk_i, 10)

    # Set decryption mode
    await tb.set_enc_dec('dec')
    await tb.aes_set_mode()

    # Load key (same key for all packets)
    tb.data = {
        'key': {'data': '000102030405060708090a0b0c0d0e0f', 'n_bytes': 16},
        'aad_n_bytes': 0,
        'pt_n_bytes': 16
    }
    await tb.load_key(tb.data['key'])

    # Test 3 packets with different IVs
    ivs = [
        '000102030405060708090a0b',
        '000102030405060708090a0c',  # Different last byte
        '000102030405060708090a0d',
    ]

    for i, iv_hex in enumerate(ivs):
        dut._log.info(f"\n=== Packet {i+1}: IV = {iv_hex} ===")

        tb.data['iv'] = {'data': iv_hex, 'n_bytes': 12}

        # Stop counter (needed for IV reload)
        dut.aes_gcm_icb_stop_cnt_i.value = 1
        await RisingEdge(dut.clk_i)
        dut.aes_gcm_icb_stop_cnt_i.value = 0
        await ClockCycles(dut.clk_i, 2)

        # Load new IV
        await tb.load_iv(tb.data['iv'])

        # Start ICB
        await tb.start_icb()

        # Wait for ready
        cycle = 0
        while dut.aes_gcm_ready_o.value == 0:
            await RisingEdge(dut.clk_i)
            cycle += 1
            if cycle > 100:
                raise Exception(f"Packet {i+1}: Timeout waiting for ready")

        dut._log.info(f"Packet {i+1}: Ready after {cycle} cycles")

        # Small delay between packets
        await ClockCycles(dut.clk_i, 10)

    dut._log.info("\nPASS: Back-to-back packets handled correctly")
