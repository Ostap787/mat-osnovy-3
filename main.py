import math
import random
import datetime
import time
import matplotlib.pyplot as plt


def generate_signal(N):
    random.seed(datetime.datetime.now().timestamp())
    return [random.random() for _ in range(N)]


def dft(signal):
    N = len(signal)
    A, B = [], []
    add_ops, mul_ops = 0, 0

    start_add_time, start_mul_time = time.time(), time.time()
    add_time, mul_time = 0, 0

    for k in range(N):
        a = 0
        b = 0
        for i in range(N):
            start_mul_time = time.time()
            real_part = signal[i] * math.cos(2 * math.pi * k * i / N)
            imag_part = signal[i] * math.sin(2 * math.pi * k * i / N)
            mul_time += time.time() - start_mul_time
            mul_ops += 2  # дві операції множення на кожну ітерацію i

            start_add_time = time.time()
            a += real_part
            b += imag_part
            add_time += time.time() - start_add_time
            add_ops += 2  # дві операції додавання

        A.append(a / N)
        B.append(b / N)

    return A, B, add_ops, mul_ops, add_time, mul_time


def fft(signal):
    N = len(signal)
    if N <= 1:
        return signal, [0], 0, 0, 0, 0

    even_signal = signal[0::2]
    odd_signal = signal[1::2]

    even_A, even_B, even_add_ops, even_mul_ops, even_add_time, even_mul_time = fft(even_signal)
    odd_A, odd_B, odd_add_ops, odd_mul_ops, odd_add_time, odd_mul_time = fft(odd_signal)

    A, B = [0] * N, [0] * N
    add_ops, mul_ops = even_add_ops + odd_add_ops, even_mul_ops + odd_mul_ops
    add_time, mul_time = even_add_time + odd_add_time, even_mul_time + odd_mul_time

    for k in range(N // 2):
        start_mul_time = time.time()
        twiddle_real = math.cos(-2 * math.pi * k / N)
        twiddle_imag = math.sin(-2 * math.pi * k / N)
        mul_time += time.time() - start_mul_time
        mul_ops += 2  # для обчислення коефіцієнтів twiddle

        start_mul_time = time.time()
        real_part = odd_A[k] * twiddle_real - odd_B[k] * twiddle_imag
        imag_part = odd_A[k] * twiddle_imag + odd_B[k] * twiddle_real
        mul_time += time.time() - start_mul_time
        mul_ops += 4  # 4 операції множення для комплексних чисел

        start_add_time = time.time()
        A[k] = even_A[k] + real_part
        B[k] = even_B[k] + imag_part
        A[k + N // 2] = even_A[k] - real_part
        B[k + N // 2] = even_B[k] - imag_part
        add_time += time.time() - start_add_time
        add_ops += 4  # 4 операції додавання

    return A, B, add_ops, mul_ops, add_time, mul_time


def run_and_measure(N_values):
    dft_add_times, dft_mul_times, fft_add_times, fft_mul_times = [], [], [], []
    dft_add_ops, dft_mul_ops, fft_add_ops, fft_mul_ops = [], [], [], []

    for N in N_values:
        signal = generate_signal(N)

        # DFT
        dft_A, dft_B, dft_add_ops_res, dft_mul_ops_res, dft_add_time, dft_mul_time = dft(signal)
        dft_add_ops.append(dft_add_ops_res)
        dft_mul_ops.append(dft_mul_ops_res)
        dft_add_times.append(dft_add_time)
        dft_mul_times.append(dft_mul_time)

        # FFT
        fft_A, fft_B, fft_add_ops_res, fft_mul_ops_res, fft_add_time, fft_mul_time = fft(signal)
        fft_add_ops.append(fft_add_ops_res)
        fft_mul_ops.append(fft_mul_ops_res)
        fft_add_times.append(fft_add_time)
        fft_mul_times.append(fft_mul_time)

        print(f"\nN = {N}:")
        print(f"DFT: Additions = {dft_add_ops_res}, Add Time = {dft_add_time:.6f}s, Multiplications = {dft_mul_ops_res}, Mul Time = {dft_mul_time:.6f}s")
        print(f"FFT: Additions = {fft_add_ops_res}, Add Time = {fft_add_time:.6f}s, Multiplications = {fft_mul_ops_res},  Mul Time = {fft_mul_time:.6f}s")

    return dft_add_times, dft_mul_times, fft_add_times, fft_mul_times, dft_add_ops, dft_mul_ops, fft_add_ops, fft_mul_ops


def plot_results(N_values, dft_add_times, dft_mul_times, fft_add_times, fft_mul_times, dft_add_ops, dft_mul_ops, fft_add_ops, fft_mul_ops):
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 2, 1)
    plt.plot(N_values, dft_add_times, label='DFT Add Time', marker='o')
    plt.plot(N_values, fft_add_times, label='FFT Add Time', marker='o')
    plt.xlabel('N (Signal Length)')
    plt.ylabel('Time (seconds)')
    plt.yscale('log')
    plt.title('Addition Time for DFT and FFT')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(N_values, dft_mul_times, label='DFT Mul Time', marker='o')
    plt.plot(N_values, fft_mul_times, label='FFT Mul Time', marker='o')
    plt.xlabel('N (Signal Length)')
    plt.ylabel('Time (seconds)')
    plt.yscale('log')
    plt.title('Multiplication Time for DFT and FFT')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(N_values, dft_add_ops, label='DFT Add Ops', marker='o')
    plt.plot(N_values, fft_add_ops, label='FFT Add Ops', marker='o')
    plt.xlabel('N (Signal Length)')
    plt.ylabel('Operations Count')
    plt.yscale('log')
    plt.title('Addition Operations Count for DFT and FFT')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(N_values, dft_mul_ops, label='DFT Mul Ops', marker='o')
    plt.plot(N_values, fft_mul_ops, label='FFT Mul Ops', marker='o')
    plt.xlabel('N (Signal Length)')
    plt.ylabel('Operations Count')
    plt.yscale('log')
    plt.title('Multiplication Operations Count for DFT and FFT')
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    N_values = [2**i for i in range(1, 12)]
    dft_add_times, dft_mul_times, fft_add_times, fft_mul_times, dft_add_ops, dft_mul_ops, fft_add_ops, fft_mul_ops = run_and_measure(N_values)
    plot_results(N_values, dft_add_times, dft_mul_times, fft_add_times, fft_mul_times, dft_add_ops, dft_mul_ops, fft_add_ops, fft_mul_ops)


if __name__ == "__main__":
    main()