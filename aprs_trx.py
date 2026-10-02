#!/usr/bin/env python3
"""
AFSK APRS Packet Transmitter via ADALM PlutoSDR
Raspberry Pi 5 edition
Requires: pyadi-iio, numpy, scipy
"""
 
import numpy as np
from scipy.signal import resample_poly
import adi
import time
from math import gcd
 
# ===================== APRS / AX.25 PARAMETERS =====================
CALLSIGN_SRC = 'NOCALL'
SSID_SRC     = 0
CALLSIGN_DST = 'APRS'
SSID_DST     = 0
APRS_INFO    = '!4807.38N/01131.00E>Test APRS via PlutoSDR /A=000000'
 
# ===================== SAMPLE RATE PARAMETERS =====================
FS_AUDIO  = 48_000
FS_PLUTO  = 576_000      # 12 x 48000, satisfies >= 521000 minimum
 
# ===================== AFSK PARAMETERS =====================
MARK_FREQ       = 1200
SPACE_FREQ      = 2200
BAUD_RATE       = 1200
SAMPLES_PER_BIT = FS_AUDIO // BAUD_RATE
FC_TX           = 144_800_000   # 144.800 MHz Europe / 144_390_000 NA
TX_GAIN         = -10           # dB, range -89 to 0
 
def encode_ax25_addr(callsign, ssid, is_last):
    callsign  = callsign.upper().ljust(6)[:6]
    addr      = [ord(c) << 1 for c in callsign]
    ssid_byte = 0x60 | ((ssid & 0x0F) << 1)
    if is_last:
        ssid_byte |= 0x01
    addr.append(ssid_byte)
    return addr
 
def compute_crc16(data):
    crc  = 0xFFFF
    poly = 0x8408
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ poly
            else:
                crc >>= 1
    return crc ^ 0xFFFF
 
def bytes_to_bits_lsb(data):
    bits = []
    for byte in data:
        for k in range(8):
            bits.append((byte >> k) & 1)
    return bits
 
def bit_stuff(bits):
    stuffed = []
    count   = 0
    for b in bits:
        stuffed.append(b)
        if b == 1:
            count += 1
            if count == 5:
                stuffed.append(0)
                count = 0
        else:
            count = 0
    return stuffed
 
def nrzi_encode(bits):
    nrzi    = []
    current = 1
    for b in bits:
        if b == 0:
            current ^= 1
        nrzi.append(current)
    return nrzi
 
def build_ax25_frame(dst_call, dst_ssid, src_call, src_ssid, info):
    dst_bytes  = encode_ax25_addr(dst_call, dst_ssid, False)
    src_bytes  = encode_ax25_addr(src_call, src_ssid, True)
    control    = [0x03]
    pid        = [0xF0]
    info_bytes = list(info.encode('ascii'))
    frame_data = dst_bytes + src_bytes + control + pid + info_bytes
    crc        = compute_crc16(frame_data)
    full_frame = frame_data + [crc & 0xFF, (crc >> 8) & 0xFF]
    bits = bytes_to_bits_lsb(full_frame)
    bits = bit_stuff(bits)
    return bits
 
def afsk_modulate(bits, f_mark, f_space, fs, spb):
    spb   = int(round(spb))
    audio = np.zeros(len(bits) * spb, dtype=np.float64)
    phase = 0.0
    t     = np.arange(spb) / fs
    for i, bit in enumerate(bits):
        freq  = f_mark if bit == 1 else f_space
        chunk = np.sin(2 * np.pi * freq * t + phase)
        audio[i*spb:(i+1)*spb] = chunk
        phase = (phase + 2 * np.pi * freq * spb / fs) % (2 * np.pi)
    return audio
 
def upsample(audio, fs_in, fs_out):
    g    = gcd(fs_out, fs_in)
    up   = fs_out // g
    down = fs_in  // g
    print(f'  Resampling {fs_in} Hz -> {fs_out} Hz  (P={up}, Q={down})')
    return resample_poly(audio, up, down)
 
def detect_pluto():
    candidates = ['ip:192.168.2.1', 'ip:pluto.local', 'usb:0']
    for uri in candidates:
        try:
            print(f'  Trying {uri} ...', end=' ', flush=True)
            sdr = adi.Pluto(uri=uri)
            print('Connected!')
            return sdr, uri
        except Exception as e:
            print(f'Failed: {e}')
    raise RuntimeError(
        'Could not find PlutoSDR.\n'
        'Check: lsusb | grep 0456\n'
        'Check: ping 192.168.2.1'
    )
 
def main():
    print('=== AFSK APRS Transmitter - ADALM PlutoSDR / Raspberry Pi 5 ===\n')
 
    print('Building AX.25 frame...')
    frame_bits = build_ax25_frame(
        CALLSIGN_DST, SSID_DST,
        CALLSIGN_SRC, SSID_SRC,
        APRS_INFO
    )
    nrzi_bits = nrzi_encode(frame_bits)
    print(f'  {len(frame_bits)} bits after stuffing')
 
    print('Modulating AFSK...')
    flag_bits  = [0,1,1,1,1,1,1,0] * 30
    tail_bits  = [0,1,1,1,1,1,1,0] * 5
    pre_audio  = afsk_modulate(nrzi_encode(flag_bits),  MARK_FREQ, SPACE_FREQ, FS_AUDIO, SAMPLES_PER_BIT)
    main_audio = afsk_modulate(nrzi_bits,                MARK_FREQ, SPACE_FREQ, FS_AUDIO, SAMPLES_PER_BIT)
    tail_audio = afsk_modulate(nrzi_encode(tail_bits),  MARK_FREQ, SPACE_FREQ, FS_AUDIO, SAMPLES_PER_BIT)
 
    full_audio = np.concatenate([pre_audio, main_audio, tail_audio])
    print(f'  Duration at {FS_AUDIO} Hz: {len(full_audio)/FS_AUDIO:.2f} s')
 
    full_audio = upsample(full_audio, FS_AUDIO, FS_PLUTO)
    full_audio /= np.max(np.abs(full_audio))
 
    scale     = 2**14
    iq_signal = (full_audio * scale).astype(np.complex64)
    print(f'  Final samples at {FS_PLUTO} Hz: {len(iq_signal)} ({len(iq_signal)/FS_PLUTO:.2f} s)')
 
    print('\nSearching for PlutoSDR...')
    sdr, uri = detect_pluto()
 
    print(f'\nConfiguring TX on {FC_TX/1e6:.3f} MHz...')
    sdr.sample_rate           = int(FS_PLUTO)
    sdr.tx_lo                 = int(FC_TX)
    sdr.tx_rf_bandwidth       = 200_000
    sdr.tx_hardwaregain_chan0 = int(TX_GAIN)
    sdr.tx_cyclic_buffer      = True
 
    print(f'  Sample rate: {sdr.sample_rate} Hz')
    print(f'  TX LO:       {sdr.tx_lo} Hz')
    print(f'  TX gain:     {sdr.tx_hardwaregain_chan0} dB')
 
    tx_duration = len(iq_signal) / FS_PLUTO
    num_repeats = 3
    total_time  = tx_duration * num_repeats
 
    print(f'\nTransmitting on {FC_TX/1e6:.3f} MHz...')
    print(f'  Cycling for {total_time:.1f} s ({num_repeats} passes)...')
 
    sdr.tx(iq_signal)
    time.sleep(total_time)
 
    sdr.tx_destroy_buffer()
    print('\nDone.')
 
if __name__ == '__main__':
    main()
