#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

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
    for (int64_t i = 0; i < num_particles; i++) {
        Drift_single_particle(part, i, length);
    }
}

void Drift_single_particle_expanded(
    int64_t num_particles,
    double* restrict rpp,
    double* restrict rvv,
    double* restrict px,
    double* restrict py,
    double* restrict x,
    double* restrict y,
    double* restrict s,
    double* restrict zeta,
    double length)
{
    // Tell compiler these pointers are aligned (adjust to your system's vector width)
    rpp   = __builtin_assume_aligned(rpp, 32);
    rvv   = __builtin_assume_aligned(rvv, 32);
    px    = __builtin_assume_aligned(px, 32);
    py    = __builtin_assume_aligned(py, 32);
    x     = __builtin_assume_aligned(x, 32);
    y     = __builtin_assume_aligned(y, 32);
    s     = __builtin_assume_aligned(s, 32);
    zeta  = __builtin_assume_aligned(zeta, 32);

    for (int64_t i = 0; i < num_particles; i++) {
        double rpp_i = rpp[i];
        double rv0v = 1.0 / rvv[i];
        double xp = px[i] * rpp_i;
        double yp = py[i] * rpp_i;
        double dzeta = 1.0 - rv0v * (1.0 + (xp * xp + yp * yp) / 2.0);

        x[i] += xp * length;
        y[i] += yp * length;
        s[i] += length;
        zeta[i] += length * dzeta;
    }
}

// Helper function to initialize particles
void init_particles(LocalParticle* restrict part, int64_t num_particles) {
    part->num_particles = num_particles;
    int64_t size = num_particles * sizeof(double);
    part->x = (double*)aligned_alloc(32, size);
    part->y = (double*)aligned_alloc(32, size);
    part->px = (double*)aligned_alloc(32, size);
    part->py = (double*)aligned_alloc(32, size);
    part->rpp = (double*)aligned_alloc(32, size);
    part->rvv = (double*)aligned_alloc(32, size);
    part->s = (double*)aligned_alloc(32, size);
    part->zeta = (double*)aligned_alloc(32, size);

    // couldn't vectorize -> no vectype for stmt: *_11 = 0.0;
    for (int64_t i = 0; i < num_particles; i++) {
        part->x[i] = 0.0;
        part->y[i] = 0.0;
        part->px[i] = (double)i * 0.001;
        part->py[i] = (double)i * 0.002;
        part->rpp[i] = 1.0 + (double)i * 0.0001;
        part->rvv[i] = 1.0 + (double)i * 0.0002;
        part->s[i]    = 0.0;
        part->zeta[i] = 0.0;
    }
}

// Main function to run the test
int main() {
    // 2^24 particles
    int64_t num_particles = 16777216;
    double length = 1.0;

    LocalParticle part;
    init_particles(&part, num_particles);

    // Run the function that should be vectorized
    clock_t begin = clock();
    Drift_track_local_particle(&part, length);
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    // Print some values to check correctness
    printf("x[0] = %f, x[1] = %f, x[2] = %f\n", part.x[0], part.x[1], part.x[2]);
    printf("Time spent = %f\n", time_spent);
    // Free allocated memory
    free(part.x);
    free(part.y);
    free(part.px);
    free(part.py);
    free(part.rpp);
    free(part.rvv);

    return 0;
}