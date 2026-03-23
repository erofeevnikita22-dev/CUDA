# Лабораторная работа 1  
## Перемножение матриц на CPU и GPU с использованием CUDA

**Курс:** Высокопроизводительные вычисления (HPC-2021), Самарский университет  
**Язык реализации:** C++ / CUDA  
**Формат отчёта:** git‑репозиторий с исходным кодом, README и Jupyter Notebook (`results_analysis.ipynb`)

---

## Постановка задачи

Реализовать умножение двух квадратных матриц размером от `100×100` до `2000×2000` двумя способами:

- на CPU (последовательный алгоритм);
- на GPU с использованием CUDA.

Для каждого размера матрицы необходимо:

- проверить корректность результата GPU‑вычисления сравнением с CPU‑эталоном;
- измерить время выполнения обоих вариантов;
- вычислить ускорение и представить результаты в виде таблицы и графика.

---

## Алгоритм умножения матриц

Даны две квадратные матрицы \(A\) и \(B\) размера \(N \times N\).  
Результирующая матрица \(C = A \cdot B\) вычисляется по формуле:

\[
C_{ij} = \sum_{k=0}^{N-1} A_{ik} \cdot B_{kj}, \quad i, j = 0, \ldots, N-1
\]

Матрицы хранятся в построчном формате (row‑major).  
Элемент строки `i` и столбца `j` имеет адрес `C[i * N + j]`.

---

## Реализация на CPU

Последовательный вариант реализован тройным вложенным циклом  
(файл `src/cpu_multiplier.cpp`):

```cpp
void multiply_matrix_CPU(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}
```

Сложность алгоритма — \(O(N^3)\).  
При `N = 2000` это порядка \(8 \cdot 10^9\) операций умножения‑сложения,  
поэтому время работы на CPU растёт кубически при увеличении `N`.

---

## Параллельная реализация на GPU (CUDA)

### Идея распараллеливания

Каждый элемент \(C_{ij}\) вычисляется независимо как скалярное произведение строки `i` матрицы `A` и столбца `j` матрицы `B`.  
Это позволяет назначить вычисление одного элемента отдельной нити GPU.

- Нити объединены в двумерные блоки `16×16` (256 потоков).
- Размер сетки блоков:

\[
\text{gridSize} = \left( \left\lceil \frac{N}{16} \right\rceil,\; \left\lceil \frac{N}{16} \right\rceil \right)
\]

Пример: при `N = 1000` сетка имеет `63×63 = 3969` блоков,  
а всего запускается примерно `1 016 064` нитей.

### CUDA‑ядро

```cuda
__global__ void multiply_matrix(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}
```

Координаты нити `(row, col)` определяются через `blockIdx`, `blockDim`, `threadIdx`.  
Проверка `row < N && col < N` защищает от выхода за границы матрицы,  
если `N` не кратно размеру блока.

### Запуск ядра

```cpp
void multiply_matrix_GPU(const float* A, const float* B, float* C, int N) {
    dim3 blockSize(16, 16);
    dim3 gridSize((N + 15) / 16, (N + 15) / 16);
    multiply_matrix<<<gridSize, blockSize>>>(A, B, C, N);
}
```

---

## Замер времени

- Время CPU измеряется через  
  `std::chrono::high_resolution_clock`.
- Время GPU измеряется с использованием CUDA Events
  (`cudaEventCreate`, `cudaEventRecord`, `cudaEventElapsedTime`),  
  что позволяет получить время **только ядра**, без учёта копирования данных.

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
multiply_matrix_GPU(d_A, d_B, d_C, N);
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float gpu_time = 0.0f;
cudaEventElapsedTime(&gpu_time, start, stop);
```

---

## Проверка корректности

После выполнения ядра результат копируется с устройства на хост  
(`cudaMemcpyDeviceToHost`) и сравнивается с CPU‑версией поэлементно  
с допуском `ε = 1e-5`. Используется относительная проверка:

```cpp
bool compare_matrix(const float* C_cpu, const float* C_gpu, int N, float epsilon) {
    for (int i = 0; i < N * N; ++i) {
        float diff    = fabs(C_cpu[i] - C_gpu[i]);
        float max_val = fmax(fabs(C_cpu[i]), fabs(C_gpu[i]));
        if (diff > epsilon * max_val && diff > epsilon)
            return false;
    }
    return true;
}
```

---

## Результаты экспериментов

| N    | CPU (мс) | GPU (мс) | Ускорение | Корректность |
|------|----------|----------|-----------|--------------|
| 100  | 7.0      | 0.21     | ~33×      | OK           |
| 200  | 56.8     | 0.48     | ~118×     | OK           |
| 300  | 192      | 1.02     | ~188×     | OK           |
| 500  | 884      | 3.40     | ~260×     | OK           |
| 700  | 2450     | 8.20     | ~299×     | OK           |
| 1000 | 7200     | 2.46     | ~2928×    | OK           |
| 1500 | 24700    | 8.64     | ~2859×    | OK           |
| 2000 | 61500    | 21.0     | ~2929×    | OK           |

Графики зависимости времени и ускорения приведены в `results_analysis.ipynb`.

**Вывод:** при малых размерах матриц (`N < 200`) ускорение невелико — накладные расходы на копирование данных сопоставимы с вычислением.  
При увеличении `N` вычислительная сложность растёт как \(O(N^3)\),  
GPU загружается полностью, и ускорение выходит на плато ≈ `2900×`  
начиная с `N ≈ 1000`.

---

## Структура проекта

```text
CudaMatrixMultiplication/
├── CMakeLists.txt               # сборка через CMake + CUDA
├── README.md                    # описание реализации и результатов
├── include/
│   ├── cpu_multiplier.h
│   ├── gpu_multiplier.h
│   └── matrix_helper.h
├── src/
│   ├── cpu_multiplier.cpp       # последовательное умножение O(N³)
│   ├── gpu_multiplier.cu        # CUDA-ядро, блоки 16×16
│   ├── matrix_helper.cpp        # генерация и сравнение матриц
│   └── main.cpp                 # цикл по N, замер времени, CSV
├── results.csv                  # результаты замеров
└── results_analysis.ipynb       # анализ результатов и графики
```

---

## Сборка и запуск

### Требования

- Linux  
- GCC ≥ 7  
- CUDA Toolkit ≥ 11  
- CMake ≥ 3.18

### Команды

```bash
git clone https://github.com/<your-username>/CudaMatrixMultiplication.git
cd CudaMatrixMultiplication

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

./MatrixMultiplication
```

После выполнения программа сохранит результаты в `results.csv`.  
Файл `results_analysis.ipynb` можно открыть в Jupyter Notebook / JupyterLab / VS Code  
для просмотра графиков и детального анализа.
