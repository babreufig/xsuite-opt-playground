#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <x86intrin.h>

// Struct with SoA layout
typedef struct {
    double* restrict x;
    double* restrict y;
    double* restrict px;
    double* restrict py;
    double* restrict rpp;
    double* restrict rvv;
    double* restrict s;
    double* restrict zeta;
    int64_t num_particles;
} LocalParticle;

//Single particle drift
static inline void Drift_single_particle(LocalParticle* restrict part, int64_t i, double length) {
    double rpp = part->rpp[i];
    double rv0v = 1.0 / part->rvv[i];
    double xp = part->px[i] * rpp;
    double yp = part->py[i] * rpp;
    double dzeta = 1 - rv0v * (1.0 + (xp * xp + yp * yp) / 2.0);

    part->x[i] += xp * length;
    part->y[i] += yp * length;
    part->s[i] += length; // using this denies vectorization if not restricted
    part->zeta[i] += length * dzeta; // using this denies vectorization
}

// Apply drift to all particles (vectorize!!)
void Drift_track_local_particle(LocalParticle* restrict part, double length) {
    int64_t num_particles = part->num_particles;

    // Vectorization expected here
    #pragma GCC ivdep
    for (int64_t i = 0; i < num_particles; i++) {
        Drift_single_particle(part, i, length);
    }
}

void Drift_track_local_particle_avx(LocalParticle* restrict part, double length) {
    int64_t num_particles = part->num_particles;

    for (int64_t i = 0; i < num_particles; i += 4) {
        __m256d rpp = _mm256_load_pd(&part->rpp[i]);
        __m256d rvv = _mm256_load_pd(&part->rvv[i]);
        __m256d px = _mm256_load_pd(&part->px[i]);
        __m256d py = _mm256_load_pd(&part->py[i]);

        __m256d xp = _mm256_mul_pd(px, rpp);
        __m256d yp = _mm256_mul_pd(py, rpp);
        __m256d rv0v = _mm256_div_pd(_mm256_set1_pd(1.0), rvv);
        __m256d temp = _mm256_fmadd_pd(xp, xp, _mm256_mul_pd(yp, yp));
        __m256d dzeta = _mm256_fnmadd_pd(rv0v, _mm256_fmadd_pd(temp, _mm256_set1_pd(0.5), _mm256_set1_pd(1.0)), _mm256_set1_pd(1.0));

        __m256d len = _mm256_set1_pd(length);
        __m256d new_x = _mm256_fmadd_pd(xp, len, _mm256_load_pd(&part->x[i]));
        __m256d new_y = _mm256_fmadd_pd(yp, len, _mm256_load_pd(&part->y[i]));
        __m256d new_s = _mm256_add_pd(_mm256_load_pd(&part->s[i]), len);
        __m256d new_zeta = _mm256_fmadd_pd(len, dzeta, _mm256_load_pd(&part->zeta[i]));

        _mm256_store_pd(&part->x[i], new_x);
        _mm256_store_pd(&part->y[i], new_y);
        _mm256_store_pd(&part->s[i], new_s);
        _mm256_store_pd(&part->zeta[i], new_zeta);
    }
}

void Drift_track_local_particle_unrolled(LocalParticle* restrict part, double length) {
    int64_t num_particles = part->num_particles;
    int64_t i;

    for (i = 0; i <= num_particles - 4; i += 4) {
        Drift_single_particle(part, i, length);
        Drift_single_particle(part, i + 1, length);
        Drift_single_particle(part, i + 2, length);
        Drift_single_particle(part, i + 3, length);
    }

    for (; i < num_particles; i++) {
        Drift_single_particle(part, i, length);
    }
}

// Helper function to initialize particles
void init_particles(LocalParticle* part, int64_t num_particles) {
    part->num_particles = num_particles;
    int64_t size = num_particles * sizeof(double);
    // part->x = (double*)aligned_alloc(32, size);
    // part->y = (double*)aligned_alloc(32, size);
    // part->px = (double*)aligned_alloc(32, size);
    // part->py = (double*)aligned_alloc(32, size);
    // part->rpp = (double*)aligned_alloc(32, size);
    // part->rvv = (double*)aligned_alloc(32, size);
    // part->s = (double*)aligned_alloc(32, size);
    // part->zeta = (double*)aligned_alloc(32, size);

    double* data = (double*)aligned_alloc(64, size * 8);

    part->x = data;
    part->y = data + num_particles;
    part->px = data + num_particles * 2;
    part->py = data + num_particles * 3;
    part->rpp = data + num_particles * 4;
    part->rvv = data + num_particles * 5;
    part->s = data + num_particles * 6;
    part->zeta = data + num_particles * 7;

    for (int64_t i = 0; i < num_particles; i++) {
        part->x[i] = 0.0;
        part->y[i] = 0.0;
        part->px[i] = (double)i * 0.001;
        part->py[i] = (double)i * 0.002;
        part->rpp[i] = 1.0 + (double)i * 0.0001;
        part->rvv[i] = 1.0 + (double)i * 0.0002;
        part->s[i]    = 0.0;
        part->zeta[i] = 0.1;
    }
}

// Main function to run the test
int main() {
    // 2^24 particles
    int64_t num_particles = 16777216;
    double length = 1.0;

    LocalParticle part0, part1, part2;
    init_particles(&part0, num_particles);
    init_particles(&part1, num_particles);
    init_particles(&part2, num_particles);

    unsigned long long start, end;

    // Baseline
    start = __rdtsc();
    Drift_track_local_particle(&part0, length);
    end = __rdtsc();
    printf("Baseline Time = \t%llu Cycles\n", end - start);

    // AVX
    start = __rdtsc();
    Drift_track_local_particle_avx(&part1, length);
    end = __rdtsc();
    printf("AVX Time = \t\t%llu Cycles\n", end - start);

    // Unrolled
    start = __rdtsc();
    Drift_track_local_particle_unrolled(&part2, length);
    end = __rdtsc();
    printf("Unrolled Time = \t%llu Cycles\n", end - start);

    printf("x[0] = %f, x[1] = %f, x[2] = %f\n", part0.x[300], part1.x[300], part2.x[300]);

    free(part0.x);
    free(part1.x);
    free(part2.x);
    return 0;
}