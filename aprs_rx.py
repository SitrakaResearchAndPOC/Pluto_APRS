#!/usr/bin/env python3
"""
AFSK APRS Packet Decoder via ADALM PlutoSDR
Raspberry Pi 5 edition
Requires: pyadi-iio, numpy, scipy
"""
 
import numpy as np
from scipy.signal import resample_poly, firwin, lfilter, hilbert
from scipy.signal import butter, sosfilt
import adi
from datetime import datetime
from math import gcd
 
FC_RX           = 144_800_000
FS_PLUTO        = 576_000
FS_AUDIO        = 48_000
MARK_FREQ       = 1200
SPACE_FREQ      = 2200
BAUD_RATE       = 1200
SAMPLES_PER_BIT = FS_AUDIO // BAUD_RATE
RX_BUFFER_SIZE  = 576_000 * 2
 
def downsample(samples, fs_in, fs_out):
    g    = gcd(fs_in, fs_out)
    up   = fs_out // g
    down = fs_in  // g
    return resample_poly(samples, up, down)
 
def fm_demodulate(iq):
    conj_product = iq[1:] * np.conj(iq[:-1])
    return np.angle(conj_product)
 
def bandpass_filter(signal, f_low, f_high, fs, order=6):
    nyq = fs / 2.0
    sos = butter(order, [f_low/nyq, f_high/nyq], btype='band', output='sos')
    return sosfilt(sos, signal)
 
def tone_filter(signal, freq, fs, bw=200.0):
    return bandpass_filter(signal, freq - bw/2, freq + bw/2, fs)
 
def afsk_demodulate(audio, fs, f_mark, f_space):
    mark_filtered  = tone_filter(audio, f_mark,  fs, bw=300)
    space_filtered = tone_filter(audio, f_space, fs, bw=300)
    mark_env  = np.abs(hilbert(mark_filtered))
    space_env = np.abs(hilbert(space_filtered))
    diff    = mark_env - space_env
    spb     = fs // BAUD_RATE
    lp_taps = firwin(spb + 1, BAUD_RATE / (fs / 2))
    diff    = lfilter(lp_taps, 1.0, diff)
    return diff
 
def recover_bits(soft_bits, fs, baud):
    spb         = fs / baud
    transitions = np.where(np.diff(np.sign(soft_bits)))[0]
    if len(transitions) == 0:
        return []
    phase = transitions[0] % spb
    n     = len(soft_bits)
    bits  = []
    pos   = phase + spb / 2
    while pos < n:
        idx = int(round(pos))
        if idx < n:
            bits.append(1 if soft_bits[idx] > 0 else 0)
        pos += spb
    return bits
 
def nrzi_decode(bits):
    if not bits:
        return []
    decoded = []
    prev    = bits[0]
    for b in bits[1:]:
        decoded.append(0 if b != prev else 1)
        prev = b
    return decoded
 
def bit_destuff(bits):
    destuffed = []
    count     = 0
    i         = 0
    while i < len(bits):
        b = bits[i]
        destuffed.append(b)
        if b == 1:
            count += 1
            if count == 5:
                i     += 1
                count  = 0
        else:
            count = 0
        i += 1
    return destuffed
 
def bits_to_bytes(bits):
    data = []
    for i in range(0, len(bits) - 7, 8):
        byte = 0
        for k in range(8):
            byte |= bits[i+k] << k
        data.append(byte)
    return data
 
def verify_crc(frame_bytes):
    if len(frame_bytes) < 3:
        return False
    data   = frame_bytes[:-2]
    crc_rx = frame_bytes[-2] | (frame_bytes[-1] << 8)
    crc    = 0xFFFF
    poly   = 0x8408
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ poly
            else:
                crc >>= 1
    return (crc ^ 0xFFFF) == crc_rx
 
def decode_ax25_addr(addr_bytes):
    callsign = ''.join(chr(b >> 1) for b in addr_bytes[:6]).strip()
    ssid     = (addr_bytes[6] >> 1) & 0x0F
    return callsign, ssid
 
def decode_ax25_frame(frame_bytes):
    if len(frame_bytes) < 16:
        return None
    dst_call, dst_ssid = decode_ax25_addr(frame_bytes[0:7])
    src_call, src_ssid = decode_ax25_addr(frame_bytes[7:14])
    offset = 14
    digis  = []
    while not (frame_bytes[offset - 1] & 0x01):
        if offset + 7 > len(frame_bytes):
            break
        digi_call, digi_ssid = decode_ax25_addr(frame_bytes[offset:offset+7])
        digis.append(f'{digi_call}-{digi_ssid}' if digi_ssid else digi_call)
        offset += 7
    if offset + 2 > len(frame_bytes):
        return None
    control    = frame_bytes[offset]
    pid        = frame_bytes[offset + 1]
    offset    += 2
    info_bytes = frame_bytes[offset:-2]
    try:
        info = bytes(info_bytes).decode('ascii', errors='replace').strip()
    except Exception:
        info = repr(bytes(info_bytes))
    return {
        'src':   f'{src_call}-{src_ssid}' if src_ssid else src_call,
        'dst':   f'{dst_call}-{dst_ssid}' if dst_ssid else dst_call,
        'digis': digis,
        'ctrl':  control,
        'pid':   pid,
        'info':  info,
    }
 
def parse_aprs_position(info, has_timestamp=False):
    try:
        offset   = 8 if has_timestamp else 1
        lat_str  = info[offset:offset+8]
        lat_deg  = float(lat_str[:2])
        lat_min  = float(lat_str[2:7])
        lat_hemi = lat_str[7]
        latitude = lat_deg + lat_min / 60.0
        if lat_hemi == 'S':
            latitude = -latitude
        sym_table = info[offset+8]
        lon_str   = info[offset+9:offset+18]
        lon_deg   = float(lon_str[:3])
        lon_min   = float(lon_str[3:8])
        lon_hemi  = lon_str[8]
        longitude = lon_deg + lon_min / 60.0
        if lon_hemi == 'W':
            longitude = -longitude
        sym_code = info[offset+18] if len(info) > offset+18 else ''
        comment  = info[offset+19:] if len(info) > offset+19 else ''
        return {
            'type':      'position',
            'latitude':  round(latitude,  5),
            'longitude': round(longitude, 5),
            'sym_table': sym_table,
            'sym_code':  sym_code,
            'comment':   comment,
        }
    except Exception:
        return {'type': 'position_raw', 'raw': info}
 
def parse_aprs_info(info):
    if not info:
        return {'type': 'unknown'}
    symbol = info[0]
    if symbol in ('!', '='):
        return parse_aprs_position(info)
    if symbol in ('@', '/'):
        return parse_aprs_position(info, has_timestamp=True)
    if symbol == '>':
        return {'type': 'status', 'text': info[1:]}
    if symbol == ':' and len(info) >= 10:
        return {'type': 'message', 'to': info[1:10].strip(), 'text': info[11:]}
    if symbol in ('`', "'"):
        return {'type': 'mice', 'raw': info}
    if symbol == ';':
        return {'type': 'object', 'raw': info[1:]}
    if symbol == '_':
        return {'type': 'weather', 'raw': info[1:]}
    return {'type': 'unknown', 'raw': info}
 
def extract_frames(bits):
    flag   = [0,1,1,1,1,1,1,0]
    frames = []
    n      = len(bits)
    i      = 0
    while i < n - 8:
        if bits[i:i+8] == flag:
            start = i + 8
            j     = start
            while j < n - 8:
                if bits[j:j+8] == flag:
                    frame_bits = bits[start:j]
                    if len(frame_bits) >= 16*8 and len(frame_bits) % 8 == 0:
                        frame_bytes = bits_to_bytes(frame_bits)
                        if verify_crc(frame_bytes):
                            frames.append(frame_bytes)
                    i = j
                    break
                j += 1
            else:
                i += 1
                continue
        i += 1
    return frames
 
def print_packet(frame, aprs, timestamp):
    print('\n' + '='*60)
    print(f'  {timestamp}')
    print(f'  {frame["src"]} -> {frame["dst"]}', end='')
    if frame['digis']:
        print(f'  via {", ".join(frame["digis"])}', end='')
    print()
    print(f'  Raw info: {frame["info"]}')
    t = aprs.get('type', 'unknown')
    if t == 'position':
        print(f'  Position:  {aprs["latitude"]:.5f}, {aprs["longitude"]:.5f}')
        print(f'  Comment:   {aprs["comment"]}')
    elif t == 'status':
        print(f'  Status:    {aprs["text"]}')
    elif t == 'message':
        print(f'  Message to {aprs["to"]}: {aprs["text"]}')
    elif t == 'weather':
        print(f'  Weather:   {aprs["raw"]}')
    elif t == 'object':
        print(f'  Object:    {aprs["raw"]}')
    else:
        print(f'  Type:      {t}')
    print('='*60)
 
def main():
    print('=== AFSK APRS Decoder - ADALM PlutoSDR / Raspberry Pi 5 ===')
    print(f'    Frequency:   {FC_RX/1e6:.3f} MHz')
    print(f'    Sample rate: {FS_PLUTO/1e3:.0f} kHz')
    print(f'    Baud rate:   {BAUD_RATE} bps')
    print(f'    Tones:       {MARK_FREQ}/{SPACE_FREQ} Hz\n')
 
    sdr = adi.Pluto('ip:192.168.2.1')
    sdr.sample_rate                 = int(FS_PLUTO)
    sdr.rx_lo                       = int(FC_RX)
    sdr.rx_rf_bandwidth             = 200_000
    sdr.gain_control_mode_chan0     = 'slow_attack'
    sdr.rx_buffer_size              = RX_BUFFER_SIZE
 
    print(f'Connected')
    print(f'  RX LO:        {sdr.rx_lo/1e6:.3f} MHz')
    print(f'  Sample rate:  {sdr.sample_rate} Hz')
    print(f'\nListening for APRS packets... (Ctrl+C to stop)\n')
 
    packets_decoded = 0
 
    try:
        while True:
            iq             = sdr.rx()
            iq_down        = downsample(iq.astype(np.complex64), FS_PLUTO, FS_AUDIO)
            audio          = fm_demodulate(iq_down)
            audio          = bandpass_filter(audio, 1000, 2400, FS_AUDIO)
            soft_bits      = afsk_demodulate(audio, FS_AUDIO, MARK_FREQ, SPACE_FREQ)
            raw_bits       = recover_bits(soft_bits, FS_AUDIO, BAUD_RATE)
            if len(raw_bits) < 100:
                continue
            decoded_bits   = nrzi_decode(raw_bits)
            destuffed_bits = bit_destuff(decoded_bits)
            frames         = extract_frames(destuffed_bits)
            for frame_bytes in frames:
                ax25 = decode_ax25_frame(frame_bytes)
                if ax25 is None:
                    continue
                aprs      = parse_aprs_info(ax25['info'])
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                packets_decoded += 1
                print_packet(ax25, aprs, timestamp)
                print(f'  Total packets decoded: {packets_decoded}')
 
    except KeyboardInterrupt:
        print(f'\n\nStopped. Total packets decoded: {packets_decoded}')
 
    finally:
        del sdr
 
if __name__ == '__main__':
    main()
