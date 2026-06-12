#!/usr/bin/env python3
"""Hardware trigger generator for the 3-camera OV9281 capture rig.

Drives one GPIO pin with a steady square wave at the cameras' frame rate.
All three OV9281 boards' TRIGGER pins (+ a shared GND) are wired together to
this pin via the breadboard, so every camera exposes on the same pulse --
hardware genlock instead of audio-based sync (these cameras have no mic).

Uses pigpio's hardware_PWM (DMA-backed) rather than RPi.GPIO software timing,
which drifts too much at 120 Hz to keep three cameras' frame counts aligned
over a multi-second clip.

Requires the pigpio daemon: `sudo pigpiod` (once per boot, or systemd-enable it).

Standalone usage (run the trigger in its own terminal, then start
record_sync.py with --no-trigger in another):

  python3 trigger.py --gpio 18 --freq 120
"""
import argparse
import time

import pigpio

DEFAULT_GPIO = 18       # hardware-PWM-capable pin (BCM numbering)
DEFAULT_FREQ_HZ = 120   # match the cameras' configured frame rate
DEFAULT_DUTY_PCT = 50


def start_trigger(pi, gpio, freq_hz, duty_pct=DEFAULT_DUTY_PCT):
    """Begin a steady square wave on `gpio`. Raises if pigpio rejects it."""
    duty = int(duty_pct / 100 * 1_000_000)  # pigpio hardware_PWM duty range is 0-1,000,000
    err = pi.hardware_PWM(gpio, int(freq_hz), duty)
    if err != 0:
        raise RuntimeError(f'hardware_PWM failed (error code {err}) on GPIO{gpio}')


def stop_trigger(pi, gpio):
    pi.hardware_PWM(gpio, 0, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gpio', type=int, default=DEFAULT_GPIO,
                    help=f'BCM GPIO number wired to the cameras\' trigger pins (default {DEFAULT_GPIO})')
    ap.add_argument('--freq', type=float, default=DEFAULT_FREQ_HZ, help='trigger frequency in Hz')
    ap.add_argument('--duty', type=float, default=DEFAULT_DUTY_PCT, help='duty cycle percent')
    ap.add_argument('--duration', type=float, default=0,
                    help='seconds to run; 0 (default) runs until Ctrl-C')
    args = ap.parse_args()

    pi = pigpio.pi()
    if not pi.connected:
        raise SystemExit('Could not connect to pigpiod -- run `sudo pigpiod` first')
    try:
        start_trigger(pi, args.gpio, args.freq, args.duty)
        print(f'Triggering GPIO{args.gpio} @ {args.freq} Hz ({args.duty}% duty)... Ctrl-C to stop')
        if args.duration > 0:
            time.sleep(args.duration)
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_trigger(pi, args.gpio)
        pi.stop()


if __name__ == '__main__':
    main()
