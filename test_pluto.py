#!/usr/bin/env python3
# Save as test_pluto.py and run: python3 test_pluto.py
 
import adi
 
candidates = ['ip:192.168.2.1', 'ip:pluto.local', 'usb:0']
 
for uri in candidates:
    try:
        sdr = adi.Pluto(uri=uri)
        print(f'Connected via {uri}')
        print(f'   Sample rate:  {sdr.sample_rate} Hz')
        print(f'   TX LO:        {sdr.tx_lo} Hz')
        print(f'   RX LO:        {sdr.rx_lo} Hz')
        del sdr
        break
    except Exception as e:
        print(f'Failed {uri}: {e}')
