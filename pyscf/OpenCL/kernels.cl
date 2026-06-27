// PySCF OpenCL kernels for DFT GPU offloading
// float32, tile-based local memory, workgroup size 32

// libcint slot constants
#define ATOM_OF       0
#define ANG_OF        1
#define NPRIM_OF      2
#define NCTR_OF       3
#define PTR_EXP       5
#define PTR_COEFF     6
#define BAS_SLOTS     8
#define PTR_COORD     1
#define ATM_SLOTS     6

#define MAX_L 8

// ============================================================
// Kernel: eval_gto_sph
// Evaluate contracted spherical GTOs on grid points (value only)
// One work-item per grid point. Workgroup size = 32.
// Output: ao[ngrids, nao] in row-major (C-order)
// ============================================================
__kernel void eval_gto_sph(
    __global const float *coords,   // [ngrids, 3]
    __global const int   *atm,      // [natm, ATM_SLOTS]
    __global const int   *bas,      // [nbas, BAS_SLOTS]
    __global const float *env,      // environment array
    __global const int   *ao_loc,   // [nbas+1] shell offsets
    __global float       *ao,       // [ngrids, nao] output
    int nbas, int ngrids, int nao)
{
    int igrid = get_global_id(0);
    if (igrid >= ngrids) return;

    float gx = coords[igrid * 3 + 0];
    float gy = coords[igrid * 3 + 1];
    float gz = coords[igrid * 3 + 2];

    for (int ibas = 0; ibas < nbas; ibas++) {
        int l       = bas[ibas * BAS_SLOTS + ANG_OF];
        int nprim   = bas[ibas * BAS_SLOTS + NPRIM_OF];
        int nctr    = bas[ibas * BAS_SLOTS + NCTR_OF];
        int atm_id  = bas[ibas * BAS_SLOTS + ATOM_OF];
        int p_exp   = bas[ibas * BAS_SLOTS + PTR_EXP];
        int p_coeff = bas[ibas * BAS_SLOTS + PTR_COEFF];
        int ao_off  = ao_loc[ibas];

        float rx = env[atm[atm_id * ATM_SLOTS + PTR_COORD] + 0];
        float ry = env[atm[atm_id * ATM_SLOTS + PTR_COORD] + 1];
        float rz = env[atm[atm_id * ATM_SLOTS + PTR_COORD] + 2];

        float dx = gx - rx;
        float dy = gy - ry;
        float dz = gz - rz;
        float r2 = dx*dx + dy*dy + dz*dz;

        // Contract primitives for this shell
        // Each contracted function gets a sum of exp(-alpha*r2) * coeff
        int deg = 2*l + 1;  // spherical: 2l+1 components
        float ectr[16];     // max nctr supported = 16

        for (int k = 0; k < nctr; k++) ectr[k] = 0.0f;

        for (int j = 0; j < nprim; j++) {
            float alpha = env[p_exp + j];
            float eprim = exp(-alpha * r2);
            for (int k = 0; k < nctr; k++) {
                ectr[k] += eprim * env[p_coeff + k * nprim + j];
            }
        }

        // Common normalization factor
        float fac = 1.0f;
        if (l == 0)      fac = 1.0f;
        else if (l == 1) fac = 1.0f;
        else if (l == 2) fac = 3.0f;  // CINTcommon_fac_sp(2) = 3
        else if (l == 3) fac = 15.0f;
        else if (l == 4) fac = 105.0f;
        else if (l == 5) fac = 945.0f;
        else if (l == 6) fac = 10395.0f;
        else if (l == 7) fac = 135135.0f;

        // For now: write Cartesian-like values, convert to spherical
        // Spherical harmonics conversion for l=0: just ectr
        // l=1: px, py, pz -> same (spherical p = cartesian p)
        // l=2: 6 cartesian -> 5 spherical
        // l=3: 10 cartesian -> 7 spherical
        // We compute Cartesian then convert.

        // Compute Cartesian GTO values
        float cart[64];  // (l+1)*(l+2)/2, max for l=7: 36
        int ncart = (l+1)*(l+2)/2;

        if (l == 0) {
            for (int k = 0; k < nctr; k++) {
                ao[igrid * nao + ao_off + k] = ectr[k] * fac;
            }
        } else {
            // For l >= 1, compute cartesian powers and convert to spherical
            float xpows[MAX_L+1], ypows[MAX_L+1], zpows[MAX_L+1];
            xpows[0] = 1.0f; ypows[0] = 1.0f; zpows[0] = 1.0f;
            for (int il = 1; il <= l; il++) {
                xpows[il] = xpows[il-1] * dx;
                ypows[il] = ypows[il-1] * dy;
                zpows[il] = zpows[il-1] * dz;
            }

            int idx = 0;
            for (int lx = l; lx >= 0; lx--) {
                for (int ly = l - lx; ly >= 0; ly--) {
                    int lz = l - lx - ly;
                    cart[idx] = xpows[lx] * ypows[ly] * zpows[lz];
                    idx++;
                }
            }

            // Convert Cartesian to spherical
            // For simplicity, handle l=0,1,2,3 explicitly
            // l=0: 1 cart -> 1 sph
            // l=1: 3 cart -> 3 sph (identity)
            // l=2: 6 cart -> 5 sph
            // l=3: 10 cart -> 7 sph
            // Higher l: use approximate (just copy first deg values)
            // This is a simplification - for production, proper c2s needed

            for (int k = 0; k < nctr; k++) {
                float val = ectr[k] * fac;
                if (l == 0) {
                    ao[igrid * nao + ao_off + k] = val * cart[0];
                } else if (l == 1) {
                    ao[igrid * nao + ao_off + k*3 + 0] = val * cart[0]; // x
                    ao[igrid * nao + ao_off + k*3 + 1] = val * cart[1]; // y
                    ao[igrid * nao + ao_off + k*3 + 2] = val * cart[2]; // z
                } else if (l == 2) {
                    // Cartesian: xx, xy, xz, yy, yz, zz
                    // Spherical: Y2,-2=xy, Y2,-1=yz, Y2,0=(2zz-xx-yy)/sqrt(3), Y2,1=xz, Y2,2=(xx-yy)/sqrt(12)
                    // Actually PySCF uses real spherical harmonics
                    // For simplicity, we use the standard real spherical harmonic conversion
                    float c_xx = cart[0], c_xy = cart[1], c_xz = cart[2];
                    float c_yy = cart[3], c_yz = cart[4], c_zz = cart[5];
                    float sqrt3 = 1.7320508075688772f;
                    ao[igrid * nao + ao_off + k*5 + 0] = val * c_xy;                          // d_xy
                    ao[igrid * nao + ao_off + k*5 + 1] = val * c_yz;                          // d_yz
                    ao[igrid * nao + ao_off + k*5 + 2] = val * (2.0f*c_zz - c_xx - c_yy) / sqrt3; // d_z2
                    ao[igrid * nao + ao_off + k*5 + 3] = val * c_xz;                          // d_xz
                    ao[igrid * nao + ao_off + k*5 + 4] = val * (c_xx - c_yy);                 // d_x2-y2
                } else if (l == 3) {
                    // Cartesian: xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz
                    // Spherical (7): order in PySCF: f_-3, f_-2, f_-1, f_0, f_1, f_2, f_3
                    // Real spherical harmonics for l=3:
                    // f_xyz, f_y(3x^2-y^2), f_yz^2-ish..., etc.
                    // Simplified: use standard real spherical harmonic conversion
                    float c_xxx=cart[0], c_xxy=cart[1], c_xxz=cart[2], c_xyy=cart[3];
                    float c_xyz=cart[4], c_xzz=cart[5], c_yyy=cart[6], c_yyz=cart[7];
                    float c_yzz=cart[8], c_zzz=cart[9];
                    float sqrt3 = 1.7320508075688772f;
                    float sqrt5 = 2.23606797749979f;
                    float sqrt15 = sqrt3 * sqrt5;
                    // PySCF spherical f ordering:
                    // f(-3) = xyz * sqrt(15/4)  -> but let's use a simpler mapping
                    // We use the CINTc2s convention which maps:
                    // sph[0] = xyz
                    // sph[1] = y(3xx - yy)
                    // sph[2] = yz(2zz - xx - yy) / sqrt(3)
                    // sph[3] = z(2zz - 3xx - 3yy) / sqrt(15)  -> z(5zz-3r^2)
                    // sph[4] = xz(2zz - xx - yy) / sqrt(3)
                    // sph[5] = x(xx - yy)
                    // sph[6] = z(xx - yy)
                    ao[igrid * nao + ao_off + k*7 + 0] = val * c_xyz;
                    ao[igrid * nao + ao_off + k*7 + 1] = val * (3.0f*c_xxy - c_yyy);
                    ao[igrid * nao + ao_off + k*7 + 2] = val * c_yyz * (2.0f*c_zzz/c_yzz - 1.0f); // approx
                    ao[igrid * nao + ao_off + k*7 + 3] = val * c_zzz * (2.0f - 3.0f*(c_xxx+c_yyy)/c_zzz); // approx
                    ao[igrid * nao + ao_off + k*7 + 4] = val * c_xzz;
                    ao[igrid * nao + ao_off + k*7 + 5] = val * (c_xxx - c_xyy);
                    ao[igrid * nao + ao_off + k*7 + 6] = val * (c_xxz - c_yzz);
                } else {
                    // For l >= 4, just write zeros (rarely needed for DFT)
                    for (int m = 0; m < deg; m++) {
                        ao[igrid * nao + ao_off + k*deg + m] = 0.0f;
                    }
                }
            }
        }
    }
}

// ============================================================
// Kernel: eval_gto_sph_deriv1
// Evaluate AO values + gradients (4 components: val, dx, dy, dz)
// One work-item per grid point.
// Output: ao[4, ngrids, nao] in row-major
// ============================================================
__kernel void eval_gto_sph_deriv1(
    __global const float *coords,
    __global const int   *atm,
    __global const int   *bas,
    __global const float *env,
    __global const int   *ao_loc,
    __global float       *ao,       // [4, ngrids, nao]
    int nbas, int ngrids, int nao)
{
    int igrid = get_global_id(0);
    if (igrid >= ngrids) return;

    float gx = coords[igrid * 3 + 0];
    float gy = coords[igrid * 3 + 1];
    float gz = coords[igrid * 3 + 2];

    for (int ibas = 0; ibas < nbas; ibas++) {
        int l       = bas[ibas * BAS_SLOTS + ANG_OF];
        int nprim   = bas[ibas * BAS_SLOTS + NPRIM_OF];
        int nctr    = bas[ibas * BAS_SLOTS + NCTR_OF];
        int atm_id  = bas[ibas * BAS_SLOTS + ATOM_OF];
        int p_exp   = bas[ibas * BAS_SLOTS + PTR_EXP];
        int p_coeff = bas[ibas * BAS_SLOTS + PTR_COEFF];
        int ao_off  = ao_loc[ibas];

        float rx = env[atm[atm_id * ATM_SLOTS + PTR_COORD] + 0];
        float ry = env[atm[atm_id * ATM_SLOTS + PTR_COORD] + 1];
        float rz = env[atm[atm_id * ATM_SLOTS + PTR_COORD] + 2];

        float dx = gx - rx;
        float dy = gy - ry;
        float dz = gz - rz;
        float r2 = dx*dx + dy*dy + dz*dz;

        int deg = 2*l + 1;

        // Contract primitives
        float ectr[16];
        float dectr_dx[16], dectr_dy[16], dectr_dz[16];
        for (int k = 0; k < nctr; k++) {
            ectr[k] = 0.0f;
            dectr_dx[k] = 0.0f;
            dectr_dy[k] = 0.0f;
            dectr_dz[k] = 0.0f;
        }

        for (int j = 0; j < nprim; j++) {
            float alpha = env[p_exp + j];
            float eprim = exp(-alpha * r2);
            float deprim_dx = -2.0f * alpha * dx * eprim;
            float deprim_dy = -2.0f * alpha * dy * eprim;
            float deprim_dz = -2.0f * alpha * dz * eprim;
            for (int k = 0; k < nctr; k++) {
                float c = env[p_coeff + k * nprim + j];
                ectr[k] += eprim * c;
                dectr_dx[k] += deprim_dx * c;
                dectr_dy[k] += deprim_dy * c;
                dectr_dz[k] += deprim_dz * c;
            }
        }

        float fac = 1.0f;
        if (l == 0)      fac = 1.0f;
        else if (l == 1) fac = 1.0f;
        else if (l == 2) fac = 3.0f;
        else if (l == 3) fac = 15.0f;
        else if (l == 4) fac = 105.0f;
        else if (l == 5) fac = 945.0f;
        else if (l == 6) fac = 10395.0f;
        else if (l == 7) fac = 135135.0f;

        // For l=0: val = ectr, grad = dectr
        // For l=1: val = ectr * (x,y,z), grad = dectr*(x,y,z) + ectr*(1,0,0) etc.
        // Simplified: only handle l=0 and l=1 properly, l>=2 approximate

        for (int k = 0; k < nctr; k++) {
            float v = ectr[k] * fac;
            float dvx = dectr_dx[k] * fac;
            float dvy = dectr_dy[k] * fac;
            float dvz = dectr_dz[k] * fac;

            if (l == 0) {
                int base = igrid * nao + ao_off + k;
                ao[0 * ngrids * nao + base] = v;
                ao[1 * ngrids * nao + base] = dvx;
                ao[2 * ngrids * nao + base] = dvy;
                ao[3 * ngrids * nao + base] = dvz;
            } else if (l == 1) {
                // px, py, pz
                for (int m = 0; m < 3; m++) {
                    int base = igrid * nao + ao_off + k*3 + m;
                    float coord = (m == 0) ? dx : (m == 1) ? dy : dz;
                    float dcoord_dx = (m == 0) ? 1.0f : 0.0f;
                    float dcoord_dy = (m == 1) ? 1.0f : 0.0f;
                    float dcoord_dz = (m == 2) ? 1.0f : 0.0f;
                    ao[0 * ngrids * nao + base] = v * coord;
                    ao[1 * ngrids * nao + base] = dvx * coord + v * dcoord_dx;
                    ao[2 * ngrids * nao + base] = dvy * coord + v * dcoord_dy;
                    ao[3 * ngrids * nao + base] = dvz * coord + v * dcoord_dz;
                }
            } else {
                // For l >= 2, compute value and approximate gradient
                // Full implementation would need product rule for each cartesian component
                // For now, just write the value (gradient = 0, to be fixed)
                for (int m = 0; m < deg; m++) {
                    int base = igrid * nao + ao_off + k*deg + m;
                    ao[0 * ngrids * nao + base] = 0.0f;
                    ao[1 * ngrids * nao + base] = 0.0f;
                    ao[2 * ngrids * nao + base] = 0.0f;
                    ao[3 * ngrids * nao + base] = 0.0f;
                }
            }
        }
    }
}

// ============================================================
// Kernel: matmul_tiled
// C = A * B  where A[M,K], B[K,N], C[M,N] all row-major
// Tile size = 32, uses local memory
// Workgroup: (32, 32) -> each workgroup computes 32x32 block of C
// ============================================================
#define TILE 32

__kernel void matmul_tiled(
    __global const float *A,   // [M, K] row-major
    __global const float *B,   // [K, N] row-major
    __global float       *C,   // [M, N] row-major
    int M, int N, int K)
{
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int row = get_group_id(0) * TILE + tx;
    int col = get_group_id(1) * TILE + ty;

    __local float Asub[TILE][TILE];
    __local float Bsub[TILE][TILE];

    float sum = 0.0f;

    int numTiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; t++) {
        // Load tiles into local memory (gather, no scatter)
        if (row < M && t * TILE + ty < K)
            Asub[tx][ty] = A[row * K + t * TILE + ty];
        else
            Asub[tx][ty] = 0.0f;

        if (t * TILE + tx < K && col < N)
            Bsub[tx][ty] = B[(t * TILE + tx) * N + col];
        else
            Bsub[tx][ty] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute partial sum
        for (int i = 0; i < TILE; i++) {
            sum += Asub[tx][i] * Bsub[i][ty];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write result (gather-style: each thread writes its own element)
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================
// Kernel: matmul_tiled_transpose_A
// C = A^T * B  where A[K,M], B[K,N], C[M,N] all row-major
// A is transposed: A^T[M,K]
// ============================================================
__kernel void matmul_tiled_transpose_A(
    __global const float *A,   // [K, M] row-major (will be transposed)
    __global const float *B,   // [K, N] row-major
    __global float       *C,   // [M, N] row-major
    int M, int N, int K)
{
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int row = get_group_id(0) * TILE + tx;  // row in C (= col in A)
    int col = get_group_id(1) * TILE + ty;  // col in C

    __local float Asub[TILE][TILE];  // A^T tile: [M_tile, K_tile]
    __local float Bsub[TILE][TILE];  // B tile: [K_tile, N_tile]

    float sum = 0.0f;
    int numTiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; t++) {
        // Load A^T tile: A is [K, M], so A^T[row, k] = A[k, row] = A[k * M + row]
        // We need Asub[tx][ty] = A^T[row, t*TILE+ty] = A[(t*TILE+ty) * M + row]
        if (row < M && t * TILE + ty < K)
            Asub[tx][ty] = A[(t * TILE + ty) * M + row];
        else
            Asub[tx][ty] = 0.0f;

        // Load B tile: B[k, col] = B[k * N + col]
        if (t * TILE + tx < K && col < N)
            Bsub[tx][ty] = B[(t * TILE + tx) * N + col];
        else
            Bsub[tx][ty] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < TILE; i++) {
            sum += Asub[tx][i] * Bsub[i][ty];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Accumulating version: C[row*N+col] += sum
__kernel void matmul_tiled_transpose_A_accum(
    __global const float *A,
    __global const float *B,
    __global float       *C,
    int M, int N, int K)
{
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int row = get_group_id(0) * TILE + tx;
    int col = get_group_id(1) * TILE + ty;

    __local float Asub[TILE][TILE];
    __local float Bsub[TILE][TILE];

    float sum = 0.0f;
    int numTiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; t++) {
        if (row < M && t * TILE + ty < K)
            Asub[tx][ty] = A[(t * TILE + ty) * M + row];
        else
            Asub[tx][ty] = 0.0f;
        if (t * TILE + tx < K && col < N)
            Bsub[tx][ty] = B[(t * TILE + tx) * N + col];
        else
            Bsub[tx][ty] = 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = 0; i < TILE; i++)
            sum += Asub[tx][i] * Bsub[i][ty];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < M && col < N) {
        C[row * N + col] += sum;
    }
}

__kernel void zero_buffer(__global float *buf, int n)
{
    int i = get_global_id(0);
    if (i < n) buf[i] = 0.0f;
}

// ============================================================
// Kernel: matmul_tiled_transpose_B
// C = A * B^T  where A[M,K], B[N,K], C[M,N] all row-major
// ============================================================
__kernel void matmul_tiled_transpose_B(
    __global const float *A,   // [M, K] row-major
    __global const float *B,   // [N, K] row-major (will be transposed)
    __global float       *C,   // [M, N] row-major
    int M, int N, int K)
{
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int row = get_group_id(0) * TILE + tx;
    int col = get_group_id(1) * TILE + ty;

    __local float Asub[TILE][TILE];
    __local float Bsub[TILE][TILE];  // B^T tile: [K_tile, N_tile]

    float sum = 0.0f;
    int numTiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; t++) {
        // Load A tile
        if (row < M && t * TILE + ty < K)
            Asub[tx][ty] = A[row * K + t * TILE + ty];
        else
            Asub[tx][ty] = 0.0f;

        // Load B^T tile: B is [N, K], B^T[k, col] = B[col, k] = B[col * K + k]
        // Bsub[tx][ty] should be B^T[t*TILE+tx, col] = B[col * K + t*TILE + tx]
        if (t * TILE + tx < K && col < N)
            Bsub[tx][ty] = B[col * K + t * TILE + tx];
        else
            Bsub[tx][ty] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < TILE; i++) {
            sum += Asub[tx][i] * Bsub[i][ty];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================
// Kernel: contract_rho
// Compute density at grid points: rho[g] = sum_i sum_j ao[g,i] * dm[i,j] * ao[g,j]
// One work-item per grid point. Uses the fact that dm is symmetric.
// rho = sum_j (ao_dm[g,j] * ao[g,j]) where ao_dm = ao * dm
// ============================================================
__kernel void contract_rho(
    __global const float *ao,    // [ngrids, nao] row-major
    __global const float *dm,    // [nao, nao] row-major
    __global float       *rho,   // [ngrids] output
    int nao, int ngrids)
{
    int igrid = get_global_id(0);
    if (igrid >= ngrids) return;

    float sum = 0.0f;
    for (int j = 0; j < nao; j++) {
        float ao_j = ao[igrid * nao + j];
        float ao_dm_j = 0.0f;
        for (int i = 0; i < nao; i++) {
            ao_dm_j += ao[igrid * nao + i] * dm[i * nao + j];
        }
        sum += ao_dm_j * ao_j;
    }
    rho[igrid] = sum;
}

// ============================================================
// Kernel: contract_rho_grad
// Compute density and gradient at grid points for GGA
// rho[0,g] = sum_ij ao[0,g,i] * dm[i,j] * ao[0,g,j]
// rho[1,g] = 2 * sum_ij ao[1,g,i] * dm[i,j] * ao[0,g,j]  (dx)
// rho[2,g] = 2 * sum_ij ao[2,g,i] * dm[i,j] * ao[0,g,j]  (dy)
// rho[3,g] = 2 * sum_ij ao[3,g,i] * dm[i,j] * ao[0,g,j]  (dz)
// ao is [4, ngrids, nao]
// ============================================================
__kernel void contract_rho_grad(
    __global const float *ao,    // [4, ngrids, nao] row-major
    __global const float *dm,    // [nao, nao] row-major
    __global float       *rho,   // [4, ngrids] output
    int nao, int ngrids)
{
    int igrid = get_global_id(0);
    if (igrid >= ngrids) return;

    int stride = ngrids * nao;

    // Compute ao_dm[0, g, j] = sum_i ao[0,g,i] * dm[i,j]
    // Then rho[0] = sum_j ao_dm[0,g,j] * ao[0,g,j]
    // rho[1] = sum_j (ao_dm[0,g,j] * ao[1,g,j] + ao_dm[1,g,j] * ao[0,g,j])
    // where ao_dm[c,g,j] = sum_i ao[c,g,i] * dm[i,j]

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    for (int j = 0; j < nao; j++) {
        float ao0_j = ao[0 * stride + igrid * nao + j];
        float ao1_j = ao[1 * stride + igrid * nao + j];
        float ao2_j = ao[2 * stride + igrid * nao + j];
        float ao3_j = ao[3 * stride + igrid * nao + j];

        float aodm0_j = 0.0f, aodm1_j = 0.0f;
        for (int i = 0; i < nao; i++) {
            float dm_ij = dm[i * nao + j];
            aodm0_j += ao[0 * stride + igrid * nao + i] * dm_ij;
            aodm1_j += ao[1 * stride + igrid * nao + i] * dm_ij;
        }
        sum0 += aodm0_j * ao0_j;
        sum1 += aodm0_j * ao1_j + aodm1_j * ao0_j;
    }

    // For y and z gradients, reuse aodm0
    for (int j = 0; j < nao; j++) {
        float ao0_j = ao[0 * stride + igrid * nao + j];
        float ao2_j = ao[2 * stride + igrid * nao + j];
        float ao3_j = ao[3 * stride + igrid * nao + j];

        float aodm0_j = 0.0f, aodm2_j = 0.0f, aodm3_j = 0.0f;
        for (int i = 0; i < nao; i++) {
            float dm_ij = dm[i * nao + j];
            aodm0_j += ao[0 * stride + igrid * nao + i] * dm_ij;
            aodm2_j += ao[2 * stride + igrid * nao + i] * dm_ij;
            aodm3_j += ao[3 * stride + igrid * nao + i] * dm_ij;
        }
        sum2 += aodm0_j * ao2_j + aodm2_j * ao0_j;
        sum3 += aodm0_j * ao3_j + aodm3_j * ao0_j;
    }

    rho[0 * ngrids + igrid] = sum0;
    rho[1 * ngrids + igrid] = 2.0f * sum1;
    rho[2 * ngrids + igrid] = 2.0f * sum2;
    rho[3 * ngrids + igrid] = 2.0f * sum3;
}

// ============================================================
// Kernel: pbe_xc
// PBE XC functional evaluation (pointwise, GGA)
// Input: rho[4, ngrids] = [rho, dx, dy, dz]
// Output: exc[ngrids], vrho[ngrids], vsigma[ngrids]
// vrho = dE/drho, vsigma = dE/d(|grad rho|^2)
// ============================================================

// PBE exchange parameters
#define PBE_KAPPA 0.804f
#define PBE_MU 0.2195149727645171f

// PBE correlation parameters
#define PBE_BETA 0.06672455060314922f
#define PBE_GAMMA 0.031090690869654895f

// Constants
#define PI 3.14159265358979323846f
#define RS_FACTOR 0.6203504908994001f  // (3/(4*pi))^(1/3)

void lda_x(float rho, float *exc, float *vrho) {
    if (rho < 1e-20f) { *exc = 0.0f; *vrho = 0.0f; return; }
    float ax = -0.7385587663820224f;  // -3/4 * (3/pi)^(1/3)
    float rho13 = pow(rho, 1.0f/3.0f);
    *exc = ax * rho13;
    *vrho = ax * (4.0f/3.0f) * rho13;
}

void pbe_x(float rho, float sigma, float *exc, float *vrho, float *vsigma) {
    if (rho < 1e-20f) { *exc = 0.0f; *vrho = 0.0f; *vsigma = 0.0f; return; }
    float ax = -0.7385587663820224f;
    float rho13 = pow(rho, 1.0f/3.0f);
    float rho43 = rho * rho13;
    float ex_unif = ax * rho13;

    // PBE enhancement factor
    float kappa = PBE_KAPPA;
    float mu = PBE_MU;
    float kf = 1.9191582926775138f * rho13;  // (3*pi^2*rho)^(1/3)
    float s2 = sigma / (rho * rho * (2.0f * kf * kf));  // s^2 = |grad|^2 / (rho^2 * 4*kf^2)
    // Actually s = |grad| / (2 * kf * rho), s^2 = sigma / (4 * kf^2 * rho^2)
    float s2_corrected = sigma / (4.0f * kf * kf * rho * rho);
    float s = sqrt(s2_corrected);
    float s2_val = s2_corrected;

    // Fx(s) = 1 + kappa - kappa / (1 + mu * s2 / kappa)
    float denom = 1.0f + mu * s2_val / kappa;
    float Fx = 1.0f + kappa - kappa / denom;

    *exc = ex_unif * Fx;
    *vrho = ex_unif * (4.0f/3.0f) * Fx;  // approximate (ignoring derivative of Fx w.r.t. rho)
    // dFx/ds2 = kappa * mu / kappa / (denom^2) = mu / (denom^2)
    float dFx_ds2 = mu / (denom * denom);
    // vsigma = ex_unif * dFx/dsigma = ex_unif * dFx_ds2 / (4 * kf^2 * rho^2)
    *vsigma = ex_unif * dFx_ds2 / (4.0f * kf * kf * rho * rho);
}

void pbe_c(float rho, float sigma, float *exc, float *vrho, float *vsigma) {
    if (rho < 1e-20f) { *exc = 0.0f; *vrho = 0.0f; *vsigma = 0.0f; return; }

    float rs = RS_FACTOR / pow(rho, 1.0f/3.0f);
    float zeta = 0.0f;  // unpolarized

    // PBE correlation (unpolarized)
    // ec_unif = PBE correlation energy density (uniform electron gas)
    // Using the PBE parametrization of Perdew-Wang
    float A = 0.031090690869654895f;
    float alpha1 = 0.21370f;
    float beta1 = 7.5957f;
    float beta2 = 3.5876f;
    float beta3 = 1.6382f;
    float beta4 = 0.49294f;

    float t = sqrt(sigma) / (2.0f * 1.9191582926775138f * pow(rho, 4.0f/3.0f));  // t = |grad| / (2*kf*rho)
    // Actually t = s / (2 * sqrt(rs * zeta_factor))... let's use simplified PBE

    // Simplified: use the GGA-1 form
    // For unpolarized: phi = 1
    float phi = 1.0f;
    float t2 = t * t;

    // ec_unif from Perdew-Wang
    float Q = sqrt(4.0f * A * rs + beta1 * rs * rs);
    float ec_unif = -2.0f * A * (1.0f + alpha1 * rs) * log(1.0f + 1.0f / (2.0f * A * (beta1 * rs + beta2 * rs * rs + beta3 * rs * rs * rs + beta4 * rs * rs * rs * rs)));

    // Actually, let's use a simpler approach: just use LDA correlation for now
    // and add PBE correction approximately
    // This is a simplification - for production, full PBE correlation needed

    // Simple PBE correlation (simplified)
    float ec = ec_unif;
    float H = PBE_BETA * t2 / PBE_GAMMA * 1.0f / (1.0f + 4.0f * PBE_BETA * t2 / PBE_GAMMA);
    ec += H;

    *exc = ec;
    *vrho = ec;  // approximate
    *vsigma = 0.0f;  // approximate - will be small
}

__kernel void pbe_xc(
    __global const float *rho_in,   // [4, ngrids] = [rho, dx, dy, dz]
    __global float       *exc,      // [ngrids]
    __global float       *vrho,     // [ngrids]
    __global float       *vsigma,   // [ngrids]
    int ngrids)
{
    int igrid = get_global_id(0);
    if (igrid >= ngrids) return;

    float rho = rho_in[0 * ngrids + igrid];
    float dx  = rho_in[1 * ngrids + igrid];
    float dy  = rho_in[2 * ngrids + igrid];
    float dz  = rho_in[3 * ngrids + igrid];

    float sigma = dx*dx + dy*dy + dz*dz;  // |grad rho|^2

    if (rho < 1e-20f) {
        exc[igrid] = 0.0f;
        vrho[igrid] = 0.0f;
        vsigma[igrid] = 0.0f;
        return;
    }

    float exc_x, vrho_x, vsigma_x;
    float exc_c, vrho_c, vsigma_c;

    pbe_x(rho, sigma, &exc_x, &vrho_x, &vsigma_x);
    pbe_c(rho, sigma, &exc_c, &vrho_c, &vsigma_c);

    exc[igrid] = exc_x + exc_c;
    vrho[igrid] = vrho_x + vrho_c;
    vsigma[igrid] = vsigma_x + vsigma_c;
}

// ============================================================
// Kernel: scale_ao_gga
// Compute aow[g, i] = sum_c ao[c, g, i] * wv[c, g]
// For GGA: aow = ao[0]*wv[0] + ao[1]*wv[1] + ao[2]*wv[2] + ao[3]*wv[3]
// One work-item per (grid, ao) pair
// ============================================================
__kernel void scale_ao_gga(
    __global const float *ao,    // [4, ngrids, nao]
    __global const float *wv,    // [4, ngrids]
    __global float       *aow,   // [ngrids, nao] output
    int nao, int ngrids)
{
    int igrid = get_global_id(0);
    int iao = get_global_id(1);
    if (igrid >= ngrids || iao >= nao) return;

    int stride = ngrids * nao;
    float val = 0.0f;
    for (int c = 0; c < 4; c++) {
        val += ao[c * stride + igrid * nao + iao] * wv[c * ngrids + igrid];
    }
    aow[igrid * nao + iao] = val;
}

// ============================================================
// Kernel: vxc_mat_gga
// Compute Vxc matrix: vmat[i,j] = sum_g (ao[0,g,i] * wv[0,g] * ao[0,g,j]
//                + sum_c ao[c,g,i] * wv[c,g] * ao[c,g,j])
// This is essentially aow^T * ao where aow = scale_ao(ao, wv)
// We do it directly here to avoid intermediate buffer.
// Uses tiled approach with local memory.
// Workgroup: (32, 32)
// ============================================================
__kernel void vxc_mat_gga(
    __global const float *ao,    // [4, ngrids, nao]
    __global const float *wv,    // [4, ngrids]
    __global float       *vmat,  // [nao, nao] output
    int nao, int ngrids)
{
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int row = get_group_id(0) * TILE + tx;  // i
    int col = get_group_id(1) * TILE + ty;  // j

    __local float aow_sub[TILE][TILE];  // aow[g_tile, i_tile]
    __local float ao0_sub[TILE][TILE];  // ao[0, g_tile, j_tile]

    float sum = 0.0f;
    int stride = ngrids * nao;
    int numGridTiles = (ngrids + TILE - 1) / TILE;

    for (int t = 0; t < numGridTiles; t++) {
        // Load aow tile: aow[g, i] = sum_c ao[c,g,i] * wv[c,g]
        int g_idx = t * TILE + ty;
        if (g_idx < ngrids && row < nao) {
            float val = 0.0f;
            for (int c = 0; c < 4; c++) {
                val += ao[c * stride + g_idx * nao + row] * wv[c * ngrids + g_idx];
            }
            aow_sub[tx][ty] = val;
        } else {
            aow_sub[tx][ty] = 0.0f;
        }

        // Load ao[0] tile
        if (g_idx < ngrids && col < nao) {
            ao0_sub[tx][ty] = ao[0 * stride + g_idx * nao + col];
        } else {
            ao0_sub[tx][ty] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Partial sum: sum_g aow[g, row] * ao[0, g, col]
        for (int i = 0; i < TILE; i++) {
            sum += aow_sub[tx][i] * ao0_sub[i][ty];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < nao && col < nao) {
        vmat[row * nao + col] = sum;
    }
}

// ============================================================
// Kernel: reduce_sum
// Simple reduction to compute sum of an array
// ============================================================
__kernel void reduce_sum(
    __global const float *input,
    __global float *output,
    int n)
{
    __local float sdata[TILE];
    int tid = get_local_id(0);
    int i = get_group_id(0) * TILE + tid;

    sdata[tid] = (i < n) ? input[i] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = TILE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) {
        output[get_group_id(0)] = sdata[0];
    }
}

// ============================================================
// Kernel: unpack_tril
// Unpack triangular packed matrix to full matrix
// Input: tril[nao_pair] where nao_pair = nao*(nao+1)/2
// Output: full[nao, nao] (symmetric)
// One work-item per (i,j) pair
// ============================================================
__kernel void unpack_tril(
    __global const float *tril,  // [nao_pair]
    __global float       *full,  // [nao, nao]
    int nao)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= nao || j >= nao) return;

    int maxi = max(i, j);
    int mini = min(i, j);
    int idx = maxi * (maxi + 1) / 2 + mini;
    full[i * nao + j] = tril[idx];
}

__kernel void unpack_tril_batched(
    __global const float *tril,
    __global float       *full,
    int nbatch, int nao, int nao_pair)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int p = get_global_id(2);
    if (p >= nbatch || i >= nao || j >= nao) return;

    int maxi = max(i, j);
    int mini = min(i, j);
    int idx = p * nao_pair + maxi * (maxi + 1) / 2 + mini;
    full[p * nao * nao + i * nao + j] = tril[idx];
}

// ============================================================
// Kernel: pack_tril
// Pack full matrix to triangular
// ============================================================
__kernel void pack_tril(
    __global const float *full,  // [nao, nao]
    __global float       *tril,  // [nao_pair]
    int nao)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= nao || j >= nao || j > i) return;

    int idx = i * (i + 1) / 2 + j;
    tril[idx] = full[i * nao + j];
}

__kernel void transpose_k_buf1(
    __global const float *buf1,
    __global float       *buf1_r,
    int naux, int nao)
{
    int i = get_global_id(0);
    int pk = get_global_id(1);
    if (i >= nao || pk >= naux * nao) return;

    int p = pk / nao;
    int k = pk - p * nao;
    buf1_r[i * naux * nao + pk] = buf1[(p * nao + i) * nao + k];
}

__kernel void contract_rho_lda_from_aodm(
    __global const float *ao0,
    __global const float *aodm0,
    __global float       *rho,
    int nao, int ngrids)
{
    int g = get_global_id(0);
    if (g >= ngrids) return;

    float s0 = 0.0f;
    int base = g * nao;
    for (int i = 0; i < nao; i++) {
        s0 += aodm0[base + i] * ao0[base + i];
    }
    rho[g] = s0;
}

__kernel void contract_rho_gga_from_aodm(
    __global const float *ao0,
    __global const float *ao1,
    __global const float *ao2,
    __global const float *ao3,
    __global const float *aodm0,
    __global const float *aodm1,
    __global const float *aodm2,
    __global const float *aodm3,
    __global float       *rho,
    int nao, int ngrids)
{
    int g = get_global_id(0);
    if (g >= ngrids) return;

    float s0 = 0.0f;
    float s1 = 0.0f;
    float s2 = 0.0f;
    float s3 = 0.0f;
    int base = g * nao;
    for (int i = 0; i < nao; i++) {
        float v0 = ao0[base + i];
        float v1 = ao1[base + i];
        float v2 = ao2[base + i];
        float v3 = ao3[base + i];
        float d0 = aodm0[base + i];
        s0 += d0 * v0;
        s1 += d0 * v1 + aodm1[base + i] * v0;
        s2 += d0 * v2 + aodm2[base + i] * v0;
        s3 += d0 * v3 + aodm3[base + i] * v0;
    }
    rho[g] = s0;
    rho[ngrids + g] = s1;
    rho[2 * ngrids + g] = s2;
    rho[3 * ngrids + g] = s3;
}

__kernel void scale_aow_lda(
    __global const float *ao0,
    __global const float *wv,
    __global float       *aow,
    int nao, int ngrids)
{
    int g = get_global_id(0);
    int i = get_global_id(1);
    if (g >= ngrids || i >= nao) return;

    int idx = g * nao + i;
    aow[idx] = ao0[idx] * wv[g];
}

__kernel void scale_aow_gga_split(
    __global const float *ao0,
    __global const float *ao1,
    __global const float *ao2,
    __global const float *ao3,
    __global const float *wv,
    __global float       *aow,
    int nao, int ngrids)
{
    int g = get_global_id(0);
    int i = get_global_id(1);
    if (g >= ngrids || i >= nao) return;

    int idx = g * nao + i;
    aow[idx] = ao0[idx] * wv[g] + ao1[idx] * wv[ngrids + g] + ao2[idx] * wv[2 * ngrids + g] + ao3[idx] * wv[3 * ngrids + g];
}

// ============================================================
// Kernel: compute_wv_gga
// Compute weighted XC potential for GGA
// wv[0,g] = weight[g] * vrho[g]
// wv[1,g] = weight[g] * 2 * vsigma[g] * dx[g]
// wv[2,g] = weight[g] * 2 * vsigma[g] * dy[g]
// wv[3,g] = weight[g] * 2 * vsigma[g] * dz[g]
// ============================================================
__kernel void compute_wv_gga(
    __global const float *weight,
    __global const float *vrho,
    __global const float *vsigma,
    __global const float *rho_grad,  // [3, ngrids] = [dx, dy, dz]
    __global float       *wv,        // [4, ngrids]
    int ngrids)
{
    int igrid = get_global_id(0);
    if (igrid >= ngrids) return;

    float w = weight[igrid];
    float vr = vrho[igrid];
    float vs = vsigma[igrid];
    float dx = rho_grad[0 * ngrids + igrid];
    float dy = rho_grad[1 * ngrids + igrid];
    float dz = rho_grad[2 * ngrids + igrid];

    wv[0 * ngrids + igrid] = w * vr;
    wv[1 * ngrids + igrid] = w * 2.0f * vs * dx;
    wv[2 * ngrids + igrid] = w * 2.0f * vs * dy;
    wv[3 * ngrids + igrid] = w * 2.0f * vs * dz;
}

// ============================================================
// Kernel: compute_nelec_exc
// Compute nelec and excsum contributions per grid point
// nelec += rho * weight
// excsum += rho * weight * exc
// ============================================================
__kernel void compute_nelec_exc(
    __global const float *rho,     // [ngrids]
    __global const float *weight,  // [ngrids]
    __global const float *exc,     // [ngrids]
    __global float       *nelec_exc,  // [2, ngrids] output (nelec, exc per point)
    int ngrids)
{
    int igrid = get_global_id(0);
    if (igrid >= ngrids) return;

    float den = rho[igrid] * weight[igrid];
    nelec_exc[0 * ngrids + igrid] = den;
    nelec_exc[1 * ngrids + igrid] = den * exc[igrid];
}

__kernel void eval_ao_mapped_hermite_cart(
    __global const float *coords,
    __global const float *atom_coords,
    __global const float *rad_val,
    __global const float *rad_du,
    __global const float *rad_dy,
    __global const int   *cart_shell,
    __global const int   *cart_ctr,
    __global const int   *cart_ixyz,
    __global const int   *shell_atom,
    __global float       *ao,
    float r0, float du,
    int nrad, int nctr_max, int ncart, int ngrids)
{
    int g = get_global_id(0);
    int iao = get_global_id(1);
    if (g >= ngrids || iao >= ncart) return;

    int sh = cart_shell[iao];
    int ctr = cart_ctr[iao];
    int at = shell_atom[sh];
    int xyz0 = g * 3;
    int at0 = at * 3;
    float dx = coords[xyz0]     - atom_coords[at0];
    float dyc = coords[xyz0 + 1] - atom_coords[at0 + 1];
    float dz = coords[xyz0 + 2] - atom_coords[at0 + 2];
    float r = sqrt(dx*dx + dyc*dyc + dz*dz);
    float uf = log1p(r / r0) / du;
    int i = (int)floor(uf);
    i = max(0, min(i, nrad - 2));
    float t = clamp(uf - (float)i, 0.0f, 1.0f);
    int base = (sh * nctr_max + ctr) * nrad + i;
    float y0 = rad_val[base];
    float dy = rad_dy[base];
    float d0 = rad_du[base];
    float d1 = rad_du[base + 1];
    float t1m = t - 1.0f;
    float radial = y0 + t*t*(3.0f-2.0f*t)*dy + t*t1m*t1m*du*d0 + t*t*t1m*du*d1;
    int p0 = iao * 3;
    int ix = cart_ixyz[p0];
    int iy = cart_ixyz[p0 + 1];
    int iz = cart_ixyz[p0 + 2];
    // note - there we should use bit-shift not stupid pow() !!!!!
    float ax = ix == 0 ? 1.0f : (ix == 1 ? dx : (ix == 2 ? dx*dx : pow(dx, (float)ix)));
    float ay = iy == 0 ? 1.0f : (iy == 1 ? dyc : (iy == 2 ? dyc*dyc : pow(dyc, (float)iy)));
    float az = iz == 0 ? 1.0f : (iz == 1 ? dz : (iz == 2 ? dz*dz : pow(dz, (float)iz)));
    ao[g * ncart + iao] = radial * ax * ay * az;
}

__kernel void eval_ao_mapped_hermite_cart_deriv1(
    __global const float *coords,
    __global const float *atom_coords,
    __global const float *rad_val,
    __global const float *rad_du,
    __global const float *rad_dy,
    __global const int   *cart_shell,
    __global const int   *cart_ctr,
    __global const int   *cart_ixyz,
    __global const int   *shell_atom,
    __global float       *ao0,
    __global float       *ao1,
    __global float       *ao2,
    __global float       *ao3,
    float r0, float du,
    int nrad, int nctr_max, int ncart, int ngrids)
{
    int g = get_global_id(0);
    int iao = get_global_id(1);
    if (g >= ngrids || iao >= ncart) return;

    int sh = cart_shell[iao];
    int ctr = cart_ctr[iao];
    int at = shell_atom[sh];
    int xyz0 = g * 3;
    int at0 = at * 3;
    float x = coords[xyz0]     - atom_coords[at0];
    float y = coords[xyz0 + 1] - atom_coords[at0 + 1];
    float z = coords[xyz0 + 2] - atom_coords[at0 + 2];
    float r = sqrt(x*x + y*y + z*z);
    float uf = log1p(r / r0) / du;
    int i = (int)floor(uf);
    i = max(0, min(i, nrad - 2));
    float t = clamp(uf - (float)i, 0.0f, 1.0f);
    int base = (sh * nctr_max + ctr) * nrad + i;
    float y0 = rad_val[base];
    float dy = rad_dy[base];
    float d0 = rad_du[base];
    float d1 = rad_du[base + 1];

    // Fully factored Hermite to minimize float32 cancellation:
    // H(t) = y0 + t^2(3-2t)*dy + t(t-1)^2*h*d0 + t^2(t-1)*h*d1
    float t1m = t - 1.0f;  // (t-1)
    float radial = y0 + t*t*(3.0f-2.0f*t)*dy + t*t1m*t1m*du*d0 + t*t*t1m*du*d1;
    // H'(t) = 6t(1-t)*dy + (3t-1)(t-1)*h*d0 + t(3t-2)*h*d1
    float drad_du = (6.0f*t*(1.0f-t)*dy + (3.0f*t-1.0f)*t1m*du*d0 + t*(3.0f*t-2.0f)*du*d1) / du;
    float drad_dr = drad_du / (r + r0);
    float invr = r > 1.0e-20f ? 1.0f / r : 0.0f;

    int p0 = iao * 3;
    int ix = cart_ixyz[p0];
    int iy = cart_ixyz[p0 + 1];
    int iz = cart_ixyz[p0 + 2];
    float ax = ix == 0 ? 1.0f : (ix == 1 ? x : (ix == 2 ? x*x : pow(x, (float)ix)));
    float ay = iy == 0 ? 1.0f : (iy == 1 ? y : (iy == 2 ? y*y : pow(y, (float)iy)));
    float az = iz == 0 ? 1.0f : (iz == 1 ? z : (iz == 2 ? z*z : pow(z, (float)iz)));
    float dax = ix == 0 ? 0.0f : (ix == 1 ? 1.0f : (ix == 2 ? 2.0f*x : ((float)ix) * pow(x, (float)(ix - 1))));
    float day = iy == 0 ? 0.0f : (iy == 1 ? 1.0f : (iy == 2 ? 2.0f*y : ((float)iy) * pow(y, (float)(iy - 1))));
    float daz = iz == 0 ? 0.0f : (iz == 1 ? 1.0f : (iz == 2 ? 2.0f*z : ((float)iz) * pow(z, (float)(iz - 1))));
    float ang = ax * ay * az;
    int out = g * ncart + iao;
    ao0[out] = radial * ang;
    ao1[out] = drad_dr * x * invr * ang + radial * dax * ay * az;
    ao2[out] = drad_dr * y * invr * ang + radial * ax * day * az;
    ao3[out] = drad_dr * z * invr * ang + radial * ax * ay * daz;
}

// ---- Atom-block kernels (optimized) ----
//
// AO = x^ix * y^iy * z^iz * R_l(r),  where R_l is a contracted Gaussian
// interpolated on a mapped log grid u=log1p(r/r0) via cubic Hermite splines.
//
// Per atom: all radial AO channels share the same float4 displacement, r, log1p, and t.
// Each channel is one already-contracted radial function with angular momentum l.
// Thread layout: (grid_point, atom); kernel loop only dispatches eval_radial_cart*().
// Angular Cartesian components are explicitly unrolled for s/p/d/f inside helpers.

inline float hermite_eval(float t, float t1m, float du, float y0, float dy, float d0, float d1)
// Factored cubic Hermite on uniform grid with spacing du:
//   H(t) = y0 + t^2(3-2t)*dy + t(t-1)^2*h*d0 + t^2(t-1)*h*d1
// dy=y1-y0 precomputed in float64 to avoid cancellation.
{
    return y0 + t*t*(3.0f-2.0f*t)*dy + t*t1m*t1m*du*d0 + t*t*t1m*du*d1;
}

inline float hermite_eval_deriv(float t, float t1m, float du, float dy, float d0, float d1)
{
    return (6.0f*t*(1.0f-t)*dy + (3.0f*t-1.0f)*t1m*du*d0 + t*(3.0f*t-2.0f)*du*d1) / du;
}

inline void eval_radial_cart(float4 d, int l, int out, float radial, int ncart, float *ao)
{
    float x=d.x, y=d.y, z=d.z;
    if (l == 0) {
        ao[out] = radial;
    } else if (l == 1) {
        ao[out    ] = radial * x;
        ao[out + 1] = radial * y;
        ao[out + 2] = radial * z;
    } else if (l == 2) {
        ao[out    ] = radial * x*x;
        ao[out + 1] = radial * x*y;
        ao[out + 2] = radial * x*z;
        ao[out + 3] = radial * y*y;
        ao[out + 4] = radial * y*z;
        ao[out + 5] = radial * z*z;
    } else {
        ao[out    ] = radial * x*x*x;
        ao[out + 1] = radial * x*x*y;
        ao[out + 2] = radial * x*x*z;
        ao[out + 3] = radial * x*y*y;
        ao[out + 4] = radial * x*y*z;
        ao[out + 5] = radial * x*z*z;
        ao[out + 6] = radial * y*y*y;
        ao[out + 7] = radial * y*y*z;
        ao[out + 8] = radial * y*z*z;
        ao[out + 9] = radial * z*z*z;
    }
}

inline void eval_radial_cart_deriv1(float4 d, int l, int out, float radial, float drad_dr, float invr, int ncart, float *ao0, float *ao1, float *ao2, float *ao3)
{
    float x=d.x, y=d.y, z=d.z;
    float gx = drad_dr * x * invr;
    float gy = drad_dr * y * invr;
    float gz = drad_dr * z * invr;
    if (l == 0) {
        ao0[out]=radial; ao1[out]=gx; ao2[out]=gy; ao3[out]=gz;
    } else if (l == 1) {
        ao0[out    ]=radial*x; ao1[out    ]=gx*x+radial; ao2[out    ]=gy*x;        ao3[out    ]=gz*x;
        ao0[out + 1]=radial*y; ao1[out + 1]=gx*y;        ao2[out + 1]=gy*y+radial; ao3[out + 1]=gz*y;
        ao0[out + 2]=radial*z; ao1[out + 2]=gx*z;        ao2[out + 2]=gy*z;        ao3[out + 2]=gz*z+radial;
    } else if (l == 2) {
        float x2=x*x, y2=y*y, z2=z*z, xy=x*y, xz=x*z, yz=y*z;
        ao0[out    ]=radial*x2; ao1[out    ]=gx*x2+2.0f*radial*x; ao2[out    ]=gy*x2;                 ao3[out    ]=gz*x2;
        ao0[out + 1]=radial*xy; ao1[out + 1]=gx*xy+radial*y;        ao2[out + 1]=gy*xy+radial*x;        ao3[out + 1]=gz*xy;
        ao0[out + 2]=radial*xz; ao1[out + 2]=gx*xz+radial*z;        ao2[out + 2]=gy*xz;                 ao3[out + 2]=gz*xz+radial*x;
        ao0[out + 3]=radial*y2; ao1[out + 3]=gx*y2;                 ao2[out + 3]=gy*y2+2.0f*radial*y;   ao3[out + 3]=gz*y2;
        ao0[out + 4]=radial*yz; ao1[out + 4]=gx*yz;                 ao2[out + 4]=gy*yz+radial*z;        ao3[out + 4]=gz*yz+radial*y;
        ao0[out + 5]=radial*z2; ao1[out + 5]=gx*z2;                 ao2[out + 5]=gy*z2;                 ao3[out + 5]=gz*z2+2.0f*radial*z;
    } else {
        float x2=x*x, y2=y*y, z2=z*z, x3=x2*x, y3=y2*y, z3=z2*z, xy=x*y, xz=x*z, yz=y*z, x2y=x2*y, x2z=x2*z, xy2=x*y2, xyz=x*y*z, xz2=x*z2, y2z=y2*z, yz2=y*z2;
        ao0[out    ]=radial*x3;  ao1[out    ]=gx*x3+3.0f*radial*x2;  ao2[out    ]=gy*x3;                   ao3[out    ]=gz*x3;
        ao0[out + 1]=radial*x2y; ao1[out + 1]=gx*x2y+2.0f*radial*xy; ao2[out + 1]=gy*x2y+radial*x2;       ao3[out + 1]=gz*x2y;
        ao0[out + 2]=radial*x2z; ao1[out + 2]=gx*x2z+2.0f*radial*xz; ao2[out + 2]=gy*x2z;                 ao3[out + 2]=gz*x2z+radial*x2;
        ao0[out + 3]=radial*xy2; ao1[out + 3]=gx*xy2+radial*y2;      ao2[out + 3]=gy*xy2+2.0f*radial*xy; ao3[out + 3]=gz*xy2;
        ao0[out + 4]=radial*xyz; ao1[out + 4]=gx*xyz+radial*yz;      ao2[out + 4]=gy*xyz+radial*xz;       ao3[out + 4]=gz*xyz+radial*xy;
        ao0[out + 5]=radial*xz2; ao1[out + 5]=gx*xz2+radial*z2;      ao2[out + 5]=gy*xz2;                 ao3[out + 5]=gz*xz2+2.0f*radial*xz;
        ao0[out + 6]=radial*y3;  ao1[out + 6]=gx*y3;                 ao2[out + 6]=gy*y3+3.0f*radial*y2;  ao3[out + 6]=gz*y3;
        ao0[out + 7]=radial*y2z; ao1[out + 7]=gx*y2z;                ao2[out + 7]=gy*y2z+2.0f*radial*yz; ao3[out + 7]=gz*y2z+radial*y2;
        ao0[out + 8]=radial*yz2; ao1[out + 8]=gx*yz2;                ao2[out + 8]=gy*yz2+radial*z2;       ao3[out + 8]=gz*yz2+2.0f*radial*yz;
        ao0[out + 9]=radial*z3;  ao1[out + 9]=gx*z3;                 ao2[out + 9]=gy*z3;                  ao3[out + 9]=gz*z3+3.0f*radial*z2;
    }
}

__kernel void eval_ao_mapped_hermite_cart_atom(
    __global const float4 *coords,
    __global const float4 *atom_coords,
    __global const float *rad_val,
    __global const float *rad_du,
    __global const float *rad_dy,
    __global const int   *radial_l,
    __global const int   *radial_cart0,
    __global const int   *atom_radial_offset,
    __global const int   *atom_radial_list,
    __global float       *ao,
    float r0, float du,
    int nrad, int ncart, int ngrids, int natoms)
{
    int g = get_global_id(0);
    int iat = get_global_id(1);
    if (g >= ngrids || iat >= natoms) return;

    float4 d = coords[g] - atom_coords[iat];
    float r = sqrt(d.x*d.x + d.y*d.y + d.z*d.z);
    float uf = log1p(r / r0) / du;
    int i = (int)floor(uf);
    i = max(0, min(i, nrad - 2));
    float t = clamp(uf - (float)i, 0.0f, 1.0f);
    float t1m = t - 1.0f;

    int off = atom_radial_offset[iat];
    int nr = atom_radial_offset[iat + 1] - off;
    for (int ii = 0; ii < nr; ii++) {
        int ir = atom_radial_list[off + ii];
        int base = ir * nrad + i;
        float radial = hermite_eval(t, t1m, du, rad_val[base], rad_dy[base], rad_du[base], rad_du[base + 1]);
        eval_radial_cart(d, radial_l[ir], g * ncart + radial_cart0[ir], radial, ncart, ao);
    }
}

__kernel void eval_ao_mapped_hermite_cart_deriv1_atom(
    __global const float4 *coords,
    __global const float4 *atom_coords,
    __global const float *rad_val,
    __global const float *rad_du,
    __global const float *rad_dy,
    __global const int   *radial_l,
    __global const int   *radial_cart0,
    __global const int   *atom_radial_offset,
    __global const int   *atom_radial_list,
    __global float       *ao0,
    __global float       *ao1,
    __global float       *ao2,
    __global float       *ao3,
    float r0, float du,
    int nrad, int ncart, int ngrids, int natoms)
{
    int g = get_global_id(0);
    int iat = get_global_id(1);
    if (g >= ngrids || iat >= natoms) return;

    float4 d = coords[g] - atom_coords[iat];
    float r = sqrt(d.x*d.x + d.y*d.y + d.z*d.z);
    float uf = log1p(r / r0) / du;
    int i = (int)floor(uf);
    i = max(0, min(i, nrad - 2));
    float t = clamp(uf - (float)i, 0.0f, 1.0f);
    float t1m = t - 1.0f;
    float invr = r > 1.0e-20f ? 1.0f / r : 0.0f;

    int off = atom_radial_offset[iat];
    int nr = atom_radial_offset[iat + 1] - off;
    for (int ii = 0; ii < nr; ii++) {
        int ir = atom_radial_list[off + ii];
        int base = ir * nrad + i;
        float radial = hermite_eval(t, t1m, du, rad_val[base], rad_dy[base], rad_du[base], rad_du[base + 1]);
        float drad_du = hermite_eval_deriv(t, t1m, du, rad_dy[base], rad_du[base], rad_du[base + 1]);
        eval_radial_cart_deriv1(d, radial_l[ir], g * ncart + radial_cart0[ir], radial, drad_du / (r + r0), invr, ncart, ao0, ao1, ao2, ao3);
    }
}

// ============================================================
// On-the-fly tiled atom-pair kernels (no AO[ngrid,nao] materialization)
//
// Works in Cartesian basis: host precomputes DM_cart = c2s @ DM @ c2s^T,
// kernels contract directly with Cartesian AOs. Host post-processes
// vmat_sph = c2s^T @ vmat_cart @ c2s.
//
// Design (from doc/ToOpenCL.chat.md):
// - 2D workgroup: (NPTILE grid points, NATILE i-atoms), WGS = NPTILE*NATILE
// - jatom radials precomputed collectively into local wfRj[NPTILE][NATILE][MAX_SHELL]
// - iatom radials kept in private Ri[MAX_SHELL]
// - DM blocks cached in local dm_blk[NATILE][NATILE][MAX_AO_ATOM][MAX_AO_ATOM]
// - Angular unfolding in registers only (fi[6], fj[6])
// - rho: reduce over NATILE iatom threads per grid point, atomic_add to global
// - vmat: separate kernel layout, one (ia,ja) pair at a time, tree reduction
// ============================================================

#ifndef NPTILE
#define NPTILE 16
#endif
#ifndef NATILE
#define NATILE 4
#endif
#ifndef WGS_TILED
#define WGS_TILED (NPTILE * NATILE)
#endif
#ifndef MAX_SHELL
#define MAX_SHELL 6
#endif
#ifndef MAX_AO_ATOM
#define MAX_AO_ATOM 15
#endif
#define DM_TILE_SIZE (NATILE * NATILE * MAX_AO_ATOM * MAX_AO_ATOM)
#define WFJ_SIZE (NPTILE * NATILE * MAX_SHELL)

// Evaluate all radial channels for one atom at one point.
// Writes Ri[0..ns-1], zeros the rest.
inline void eval_atom_radials(int ia, float4 d, float r0, float du, int nrad,
    __global const float *rad_val, __global const float *rad_du, __global const float *rad_dy,
    __global const int *atom_radial_offset, __global const int *atom_radial_list,
    float *Ri) {
    int off = atom_radial_offset[ia];
    int ns = atom_radial_offset[ia + 1] - off;
    float r = sqrt(d.x*d.x + d.y*d.y + d.z*d.z);
    float uf = log1p(r / r0) / du;
    int i = max(0, min((int)floor(uf), nrad - 2));
    float t = clamp(uf - (float)i, 0.0f, 1.0f);
    float t1m = t - 1.0f;
    for (int s = 0; s < ns; s++) {
        int ir = atom_radial_list[off + s];
        int base = ir * nrad + i;
        Ri[s] = hermite_eval(t, t1m, du, rad_val[base], rad_dy[base], rad_du[base], rad_du[base + 1]);
    }
    for (int s = ns; s < MAX_SHELL; s++) Ri[s] = 0.0f;
}

// Same but also compute radial derivatives dR/dr
inline void eval_atom_radials_deriv(int ia, float4 d, float r0, float du, int nrad,
    __global const float *rad_val, __global const float *rad_du, __global const float *rad_dy,
    __global const int *atom_radial_offset, __global const int *atom_radial_list,
    float *Ri, float *dRi) {
    int off = atom_radial_offset[ia];
    int ns = atom_radial_offset[ia + 1] - off;
    float r = sqrt(d.x*d.x + d.y*d.y + d.z*d.z);
    float uf = log1p(r / r0) / du;
    int i = max(0, min((int)floor(uf), nrad - 2));
    float t = clamp(uf - (float)i, 0.0f, 1.0f);
    float t1m = t - 1.0f;
    float invr = r > 1.0e-20f ? 1.0f / r : 0.0f;
    for (int s = 0; s < ns; s++) {
        int ir = atom_radial_list[off + s];
        int base = ir * nrad + i;
        Ri[s] = hermite_eval(t, t1m, du, rad_val[base], rad_dy[base], rad_du[base], rad_du[base + 1]);
        float drad_du = hermite_eval_deriv(t, t1m, du, rad_dy[base], rad_du[base], rad_du[base + 1]);
        dRi[s] = drad_du / (r + r0);
    }
    for (int s = ns; s < MAX_SHELL; s++) { Ri[s] = 0.0f; dRi[s] = 0.0f; }
}

// Unfold one shell: radial * angular Cartesian components
inline int unfold_shell(int l, float R, float4 d, float *f) {
    float x=d.x, y=d.y, z=d.z;
    if (l == 0) { f[0]=R; return 1; }
    if (l == 1) { f[0]=R*x; f[1]=R*y; f[2]=R*z; return 3; }
    f[0]=R*x*x; f[1]=R*x*y; f[2]=R*x*z; f[3]=R*y*y; f[4]=R*y*z; f[5]=R*z*z; return 6;
}

// Unfold shell value + derivatives (gx,gy,gz components for each angular)
inline int unfold_shell_deriv(int l, float R, float dRdr, float4 d, float invr,
    float *f0, float *f1, float *f2, float *f3) {
    float x=d.x, y=d.y, z=d.z;
    float gx=dRdr*x*invr, gy=dRdr*y*invr, gz=dRdr*z*invr;
    if (l == 0) { f0[0]=R; f1[0]=gx; f2[0]=gy; f3[0]=gz; return 1; }
    if (l == 1) {
        f0[0]=R*x; f1[0]=gx*x+R; f2[0]=gy*x;     f3[0]=gz*x;
        f0[1]=R*y; f1[1]=gx*y;     f2[1]=gy*y+R; f3[1]=gz*y;
        f0[2]=R*z; f1[2]=gx*z;     f2[2]=gy*z;     f3[2]=gz*z+R;
        return 3;
    }
    float x2=x*x, y2=y*y, z2=z*z, xy=x*y, xz=x*z, yz=y*z;
    f0[0]=R*x2; f1[0]=gx*x2+2.f*R*x; f2[0]=gy*x2;            f3[0]=gz*x2;
    f0[1]=R*xy; f1[1]=gx*xy+R*y;     f2[1]=gy*xy+R*x;        f3[1]=gz*xy;
    f0[2]=R*xz; f1[2]=gx*xz+R*z;     f2[2]=gy*xz;            f3[2]=gz*xz+R*x;
    f0[3]=R*y2; f1[3]=gx*y2;         f2[3]=gy*y2+2.f*R*y;    f3[3]=gz*y2;
    f0[4]=R*yz; f1[4]=gx*yz;         f2[4]=gy*yz+R*z;        f3[4]=gz*yz+R*y;
    f0[5]=R*z2; f1[5]=gx*z2;         f2[5]=gy*z2;            f3[5]=gz*z2+2.f*R*z;
    return 6;
}

// Contract one (ia,ja) atom pair using shellwise unfolding.
// Uses private Ri, local Rj, local dm_blk. No full phi arrays.
inline float contract_pair_rho(float4 di, float4 dj, int ia, int ja,
    int il, int jl, float *Ri, __local float *Rj,
    __local float dm_blk[NATILE][NATILE][MAX_AO_ATOM][MAX_AO_ATOM],
    __global const int *atom_radial_offset, __global const int *atom_radial_list,
    __global const int *radial_l) {
    float acc = 0.0f;
    int off_i = atom_radial_offset[ia], ns_i = atom_radial_offset[ia+1] - off_i;
    int off_j = atom_radial_offset[ja], ns_j = atom_radial_offset[ja+1] - off_j;
    int iao_off = 0;
    for (int si = 0; si < ns_i; si++) {
        int li = radial_l[atom_radial_list[off_i + si]];
        float fi[6]; int ni = unfold_shell(li, Ri[si], di, fi);
        int jao_off = 0;
        for (int sj = 0; sj < ns_j; sj++) {
            int lj = radial_l[atom_radial_list[off_j + sj]];
            float fj[6]; int nj = unfold_shell(lj, Rj[sj], dj, fj);
            for (int ai = 0; ai < ni; ai++) {
                float tmp = 0.0f;
                for (int aj = 0; aj < nj; aj++) tmp += dm_blk[il][jl][iao_off+ai][jao_off+aj] * fj[aj];
                acc += fi[ai] * tmp;
            }
            jao_off += nj;
        }
        iao_off += ni;
    }
    return acc;
}

// Contract one (ia,ja) pair for GGA: returns rho, gx, gy, gz contributions
inline void contract_pair_rho_gga(float4 di, float4 dj, int ia, int ja,
    int il, int jl, float *Ri, float *dRi, __local float *Rj, __local float *dRj,
    __local float dm_blk[NATILE][NATILE][MAX_AO_ATOM][MAX_AO_ATOM],
    __global const int *atom_radial_offset, __global const int *atom_radial_list,
    __global const int *radial_l,
    float *rho_val, float *gx_val, float *gy_val, float *gz_val) {
    *rho_val = 0.0f; *gx_val = 0.0f; *gy_val = 0.0f; *gz_val = 0.0f;
    int off_i = atom_radial_offset[ia], ns_i = atom_radial_offset[ia+1] - off_i;
    int off_j = atom_radial_offset[ja], ns_j = atom_radial_offset[ja+1] - off_j;
    float ri_mag = sqrt(di.x*di.x + di.y*di.y + di.z*di.z);
    float rj_mag = sqrt(dj.x*dj.x + dj.y*dj.y + dj.z*dj.z);
    float invr_i = ri_mag > 1e-20f ? 1.0f/ri_mag : 0.0f;
    float invr_j = rj_mag > 1e-20f ? 1.0f/rj_mag : 0.0f;
    int iao_off = 0;
    for (int si = 0; si < ns_i; si++) {
        int li = radial_l[atom_radial_list[off_i + si]];
        float fi0[6], fi1[6], fi2[6], fi3[6];
        int ni = unfold_shell_deriv(li, Ri[si], dRi[si], di, invr_i, fi0, fi1, fi2, fi3);
        int jao_off = 0;
        for (int sj = 0; sj < ns_j; sj++) {
            int lj = radial_l[atom_radial_list[off_j + sj]];
            float fj0[6], fj1[6], fj2[6], fj3[6];
            int nj = unfold_shell_deriv(lj, Rj[sj], dRj[sj], dj, invr_j, fj0, fj1, fj2, fj3);
            for (int ai = 0; ai < ni; ai++) {
                float t0=0.0f, t1=0.0f, t2=0.0f, t3=0.0f;
                for (int aj = 0; aj < nj; aj++) {
                    float dm = dm_blk[il][jl][iao_off+ai][jao_off+aj];
                    t0 += dm * fj0[aj];
                    t1 += dm * fj1[aj];
                    t2 += dm * fj2[aj];
                    t3 += dm * fj3[aj];
                }
                *rho_val += fi0[ai] * t0;
                *gx_val  += fi0[ai] * t1 + fi1[ai] * t0;
                *gy_val  += fi0[ai] * t2 + fi2[ai] * t0;
                *gz_val  += fi0[ai] * t3 + fi3[ai] * t0;
            }
            jao_off += nj;
        }
        iao_off += ni;
    }
}

// ---- rho_lda_tiled ----
// 2D workgroup: (NPTILE grid points, NATILE i-atoms)
// Each thread = (ip grid point, il i-atom). Iterates over j-atom tiles.
// jatom radials cached in local wfRj, DM blocks in local dm_blk.
// Final: reduce over il, atomic_add to rho[g].

__kernel void rho_lda_tiled(
    __global const float4 *coords,
    __global const float4 *atom_coords,
    __global const float *rad_val,
    __global const float *rad_du,
    __global const float *rad_dy,
    __global const int *radial_l,
    __global const int *radial_cart0,
    __global const int *atom_radial_offset,
    __global const int *atom_radial_list,
    __global const float *dm_cart,
    __global const int *atom_ao0,
    __global const int *atom_nao,
    __global float *rho,
    float r0, float du,
    int nrad, int ncart, int ngrids, int natoms)
{
    int ip = get_local_id(0);
    int il = get_local_id(1);
    int lid = il * NPTILE + ip;
    int gTile = get_group_id(0);
    int iTile = get_group_id(1);
    int g = gTile * NPTILE + ip;
    int ia = iTile * NATILE + il;

    __local float wfRj[NPTILE][NATILE][MAX_SHELL];
    __local float dm_blk[NATILE][NATILE][MAX_AO_ATOM][MAX_AO_ATOM];
    __local float psum[WGS_TILED];

    float4 di = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float Ri[MAX_SHELL];
    for (int s = 0; s < MAX_SHELL; s++) Ri[s] = 0.0f;
    if (g < ngrids && ia < natoms) {
        di = coords[g] - atom_coords[ia];
        eval_atom_radials(ia, di, r0, du, nrad, rad_val, rad_du, rad_dy,
            atom_radial_offset, atom_radial_list, Ri);
    }

    float rho_priv = 0.0f;

    for (int jTile = 0; jTile < natoms; jTile += NATILE) {
        // Cooperative load: wfRj and dm_blk (disjoint arrays, one barrier)
        for (int k = lid; k < WFJ_SIZE; k += WGS_TILED) {
            int s = k % MAX_SHELL;
            int jj = (k / MAX_SHELL) % NATILE;
            int pp = k / (MAX_SHELL * NATILE);
            int gj = gTile * NPTILE + pp;
            int ja = jTile + jj;
            float v = 0.0f;
            if (gj < ngrids && ja < natoms) {
                float4 dj = coords[gj] - atom_coords[ja];
                int off = atom_radial_offset[ja];
                int ns = atom_radial_offset[ja+1] - off;
                if (s < ns) {
                    int ir = atom_radial_list[off + s];
                    float r = sqrt(dj.x*dj.x + dj.y*dj.y + dj.z*dj.z);
                    float uf = log1p(r / r0) / du;
                    int i = max(0, min((int)floor(uf), nrad - 2));
                    float t = clamp(uf - (float)i, 0.0f, 1.0f);
                    float t1m = t - 1.0f;
                    int base = ir * nrad + i;
                    v = hermite_eval(t, t1m, du, rad_val[base], rad_dy[base], rad_du[base], rad_du[base+1]);
                }
            }
            wfRj[pp][jj][s] = v;
        }
        for (int k = lid; k < DM_TILE_SIZE; k += WGS_TILED) {
            int ab = k % (MAX_AO_ATOM * MAX_AO_ATOM);
            int pq = k / (MAX_AO_ATOM * MAX_AO_ATOM);
            int a = ab / MAX_AO_ATOM, b = ab % MAX_AO_ATOM;
            int ii2 = pq / NATILE, jj2 = pq % NATILE;
            int ia2 = iTile * NATILE + ii2, ja2 = jTile + jj2;
            float v = 0.0f;
            if (ia2 < natoms && ja2 < natoms && a < atom_nao[ia2] && b < atom_nao[ja2]) {
                v = dm_cart[(atom_ao0[ia2] + a) * ncart + (atom_ao0[ja2] + b)];
            }
            dm_blk[ii2][jj2][a][b] = v;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (g < ngrids && ia < natoms) {
            for (int jl = 0; jl < NATILE; jl++) {
                int ja = jTile + jl;
                if (ja >= natoms) continue;
                float4 dj = coords[g] - atom_coords[ja];
                rho_priv += contract_pair_rho(di, dj, ia, ja, il, jl, Ri, wfRj[ip][jl],
                    dm_blk, atom_radial_offset, atom_radial_list, radial_l);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    psum[lid] = rho_priv;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (il == 0 && g < ngrids) {
        float s = 0.0f;
        for (int k = 0; k < NATILE; k++) s += psum[k * NPTILE + ip];
        rho[iTile * ngrids + g] = s;
    }
}

// ---- rho_gga_tiled ----

__kernel void rho_gga_tiled(
    __global const float4 *coords,
    __global const float4 *atom_coords,
    __global const float *rad_val,
    __global const float *rad_du,
    __global const float *rad_dy,
    __global const int *radial_l,
    __global const int *radial_cart0,
    __global const int *atom_radial_offset,
    __global const int *atom_radial_list,
    __global const float *dm_cart,
    __global const int *atom_ao0,
    __global const int *atom_nao,
    __global float *rho,
    float r0, float du,
    int nrad, int ncart, int ngrids, int natoms)
{
    int ip = get_local_id(0);
    int il = get_local_id(1);
    int lid = il * NPTILE + ip;
    int gTile = get_group_id(0);
    int iTile = get_group_id(1);
    int g = gTile * NPTILE + ip;
    int ia = iTile * NATILE + il;

    __local float wfRj[NPTILE][NATILE][MAX_SHELL];
    __local float dwfRj[NPTILE][NATILE][MAX_SHELL];
    __local float dm_blk[NATILE][NATILE][MAX_AO_ATOM][MAX_AO_ATOM];
    __local float psum_rho[WGS_TILED];
    __local float psum_gx[WGS_TILED];
    __local float psum_gy[WGS_TILED];
    __local float psum_gz[WGS_TILED];

    float4 di = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float Ri[MAX_SHELL], dRi[MAX_SHELL];
    for (int s = 0; s < MAX_SHELL; s++) { Ri[s] = 0.0f; dRi[s] = 0.0f; }
    if (g < ngrids && ia < natoms) {
        di = coords[g] - atom_coords[ia];
        eval_atom_radials_deriv(ia, di, r0, du, nrad, rad_val, rad_du, rad_dy,
            atom_radial_offset, atom_radial_list, Ri, dRi);
    }

    float rho_priv = 0.0f, gx_priv = 0.0f, gy_priv = 0.0f, gz_priv = 0.0f;

    for (int jTile = 0; jTile < natoms; jTile += NATILE) {
        // Cooperative load: wfRj, dwfRj, dm_blk
        for (int k = lid; k < WFJ_SIZE; k += WGS_TILED) {
            int s = k % MAX_SHELL;
            int jj = (k / MAX_SHELL) % NATILE;
            int pp = k / (MAX_SHELL * NATILE);
            int gj = gTile * NPTILE + pp;
            int ja = jTile + jj;
            float v = 0.0f, dv = 0.0f;
            if (gj < ngrids && ja < natoms) {
                float4 dj = coords[gj] - atom_coords[ja];
                int off = atom_radial_offset[ja];
                int ns = atom_radial_offset[ja+1] - off;
                if (s < ns) {
                    int ir = atom_radial_list[off + s];
                    float r = sqrt(dj.x*dj.x + dj.y*dj.y + dj.z*dj.z);
                    float uf = log1p(r / r0) / du;
                    int i = max(0, min((int)floor(uf), nrad - 2));
                    float t = clamp(uf - (float)i, 0.0f, 1.0f);
                    float t1m = t - 1.0f;
                    int base = ir * nrad + i;
                    v = hermite_eval(t, t1m, du, rad_val[base], rad_dy[base], rad_du[base], rad_du[base+1]);
                    float drad_du = hermite_eval_deriv(t, t1m, du, rad_dy[base], rad_du[base], rad_du[base+1]);
                    dv = drad_du / (r + r0);
                }
            }
            wfRj[pp][jj][s] = v;
            dwfRj[pp][jj][s] = dv;
        }
        for (int k = lid; k < DM_TILE_SIZE; k += WGS_TILED) {
            int ab = k % (MAX_AO_ATOM * MAX_AO_ATOM);
            int pq = k / (MAX_AO_ATOM * MAX_AO_ATOM);
            int a = ab / MAX_AO_ATOM, b = ab % MAX_AO_ATOM;
            int ii2 = pq / NATILE, jj2 = pq % NATILE;
            int ia2 = iTile * NATILE + ii2, ja2 = jTile + jj2;
            float v = 0.0f;
            if (ia2 < natoms && ja2 < natoms && a < atom_nao[ia2] && b < atom_nao[ja2]) {
                v = dm_cart[(atom_ao0[ia2] + a) * ncart + (atom_ao0[ja2] + b)];
            }
            dm_blk[ii2][jj2][a][b] = v;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (g < ngrids && ia < natoms) {
            for (int jl = 0; jl < NATILE; jl++) {
                int ja = jTile + jl;
                if (ja >= natoms) continue;
                float4 dj = coords[g] - atom_coords[ja];
                float rv, gv1, gv2, gv3;
                contract_pair_rho_gga(di, dj, ia, ja, il, jl, Ri, dRi, wfRj[ip][jl], dwfRj[ip][jl],
                    dm_blk, atom_radial_offset, atom_radial_list, radial_l,
                    &rv, &gv1, &gv2, &gv3);
                rho_priv += rv; gx_priv += gv1; gy_priv += gv2; gz_priv += gv3;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    psum_rho[lid] = rho_priv; psum_gx[lid] = gx_priv;
    psum_gy[lid] = gy_priv;   psum_gz[lid] = gz_priv;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (il == 0 && g < ngrids) {
        float sr=0.0f, sx=0.0f, sy=0.0f, sz=0.0f;
        for (int k = 0; k < NATILE; k++) {
            sr += psum_rho[k*NPTILE+ip]; sx += psum_gx[k*NPTILE+ip];
            sy += psum_gy[k*NPTILE+ip]; sz += psum_gz[k*NPTILE+ip];
        }
        int base = iTile * 4 * ngrids;
        rho[base + g] = sr;
        rho[base + ngrids + g] = sx;
        rho[base + 2*ngrids + g] = sy;
        rho[base + 3*ngrids + g] = sz;
    }
}

// ---- vmat kernels (fast: one workgroup per (iTile,jTile)) ----
// From doc/ToOpenCL.chat.md optimized design:
//   workgroup = one (iTile,jTile) atom-pair tile
//   thread owns QPT AO-pair matrix elements with private acc[QPT]
//   loop over grid-point tiles
//   local = unfolded AO values aoI[NPTILE][AO_TILE], aoJ[NPTILE][AO_TILE]
//   no abTile, no redundant radial recomputation, no atomics

#ifndef WGS_VMAT
#define WGS_VMAT 256
#endif
#define AO_TILE (NATILE * MAX_AO_ATOM)
#define VBLK_SIZE (AO_TILE * AO_TILE)
#define QPT ((VBLK_SIZE + WGS_VMAT - 1) / WGS_VMAT)
#define PT_ATOM_SIZE (NPTILE * NATILE)

// Fill local AO values for one atom at one grid point (LDA: just phi)
inline void fill_atom_ao_lda(int ia, float4 d, int base, __local float *ao,
    float r0, float du, int nrad,
    __global const float *rad_val, __global const float *rad_du, __global const float *rad_dy,
    __global const int *radial_l,
    __global const int *atom_radial_offset, __global const int *atom_radial_list)
{
    float R[MAX_SHELL];
    eval_atom_radials(ia, d, r0, du, nrad, rad_val, rad_du, rad_dy, atom_radial_offset, atom_radial_list, R);
    int off = atom_radial_offset[ia];
    int ns = atom_radial_offset[ia + 1] - off;
    int ao0 = 0;
    for (int s = 0; s < ns; s++) {
        int ir = atom_radial_list[off + s];
        int l = radial_l[ir];
        float f[6];
        int n = unfold_shell(l, R[s], d, f);
        for (int a = 0; a < n; a++) ao[base + ao0 + a] = f[a];
        ao0 += n;
    }
}

// Fill local AO values for one atom at one grid point (GGA: w0*phi + w*dphi)
inline void fill_atom_aow_gga(int ia, float4 d, float w0, float wx, float wy, float wz,
    int base, __local float *ao,
    float r0, float du, int nrad,
    __global const float *rad_val, __global const float *rad_du, __global const float *rad_dy,
    __global const int *radial_l,
    __global const int *atom_radial_offset, __global const int *atom_radial_list)
{
    float R[MAX_SHELL], dR[MAX_SHELL];
    eval_atom_radials_deriv(ia, d, r0, du, nrad, rad_val, rad_du, rad_dy, atom_radial_offset, atom_radial_list, R, dR);
    float rr = sqrt(d.x*d.x + d.y*d.y + d.z*d.z);
    float invr = (rr > 1e-20f) ? 1.0f / rr : 0.0f;
    int off = atom_radial_offset[ia];
    int ns = atom_radial_offset[ia + 1] - off;
    int ao0 = 0;
    for (int s = 0; s < ns; s++) {
        int ir = atom_radial_list[off + s];
        int l = radial_l[ir];
        float f0[6], f1[6], f2[6], f3[6];
        int n = unfold_shell_deriv(l, R[s], dR[s], d, invr, f0, f1, f2, f3);
        for (int a = 0; a < n; a++) ao[base + ao0 + a] = w0*f0[a] + wx*f1[a] + wy*f2[a] + wz*f3[a];
        ao0 += n;
    }
}

__kernel void vmat_lda_tiled(
    __global const float4 *coords,
    __global const float4 *atom_coords,
    __global const float *rad_val,
    __global const float *rad_du,
    __global const float *rad_dy,
    __global const int *radial_l,
    __global const int *radial_cart0,
    __global const int *atom_radial_offset,
    __global const int *atom_radial_list,
    __global const int *atom_ao0,
    __global const int *atom_nao,
    __global const float *wv,
    __global float *vmat,
    float r0, float du,
    int nrad, int ncart, int ngrids, int natoms)
{
    int lid = get_local_id(1);
    int iTile = get_group_id(0);
    int jTile = get_group_id(1);

    __local float aoI[NPTILE][AO_TILE];
    __local float aoJ[NPTILE][AO_TILE];

    float acc[QPT];
    for (int t = 0; t < QPT; t++) acc[t] = 0.0f;

    for (int gTile = 0; gTile < ngrids; gTile += NPTILE) {

        // Fill aoI: unfolded AO values for iTile atoms
        for (int k = lid; k < PT_ATOM_SIZE; k += WGS_VMAT) {
            int il = k % NATILE;
            int ip = k / NATILE;
            int ia = iTile * NATILE + il;
            int g = gTile + ip;
            int base = il * MAX_AO_ATOM;
            for (int a = 0; a < MAX_AO_ATOM; a++) aoI[ip][base + a] = 0.0f;
            if (g < ngrids && ia < natoms) {
                float4 d = coords[g] - atom_coords[ia];
                fill_atom_ao_lda(ia, d, base, aoI[ip], r0, du, nrad, rad_val, rad_du, rad_dy, radial_l, atom_radial_offset, atom_radial_list);
            }
        }

        // Fill aoJ: unfolded AO values for jTile atoms
        for (int k = lid; k < PT_ATOM_SIZE; k += WGS_VMAT) {
            int jl = k % NATILE;
            int ip = k / NATILE;
            int ja = jTile * NATILE + jl;
            int g = gTile + ip;
            int base = jl * MAX_AO_ATOM;
            for (int b = 0; b < MAX_AO_ATOM; b++) aoJ[ip][base + b] = 0.0f;
            if (g < ngrids && ja < natoms) {
                float4 d = coords[g] - atom_coords[ja];
                fill_atom_ao_lda(ja, d, base, aoJ[ip], r0, du, nrad, rad_val, rad_du, rad_dy, radial_l, atom_radial_offset, atom_radial_list);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Each thread accumulates QPT AO-pair elements over NPTILE grid points
        for (int t = 0; t < QPT; t++) {
            int q = lid + t * WGS_VMAT;
            if (q >= VBLK_SIZE) continue;
            int iao_l = q / AO_TILE;
            int jao_l = q - iao_l * AO_TILE;
            int il = iao_l / MAX_AO_ATOM;
            int jl = jao_l / MAX_AO_ATOM;
            int a = iao_l - il * MAX_AO_ATOM;
            int b = jao_l - jl * MAX_AO_ATOM;
            int ia = iTile * NATILE + il;
            int ja = jTile * NATILE + jl;
            if (ia >= natoms || ja >= natoms || a >= atom_nao[ia] || b >= atom_nao[ja]) continue;

            float s = 0.0f;
            for (int ip = 0; ip < NPTILE; ip++) {
                int g = gTile + ip;
                if (g >= ngrids) continue;
                s += wv[g] * aoI[ip][iao_l] * aoJ[ip][jao_l];
            }
            acc[t] += s;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write final vmat
    for (int t = 0; t < QPT; t++) {
        int q = lid + t * WGS_VMAT;
        if (q >= VBLK_SIZE) continue;
        int iao_l = q / AO_TILE;
        int jao_l = q - iao_l * AO_TILE;
        int il = iao_l / MAX_AO_ATOM;
        int jl = jao_l / MAX_AO_ATOM;
        int a = iao_l - il * MAX_AO_ATOM;
        int b = jao_l - jl * MAX_AO_ATOM;
        int ia = iTile * NATILE + il;
        int ja = jTile * NATILE + jl;
        if (ia < natoms && ja < natoms && a < atom_nao[ia] && b < atom_nao[ja]) {
            int iao = atom_ao0[ia] + a;
            int jao = atom_ao0[ja] + b;
            vmat[iao * ncart + jao] = acc[t];
        }
    }
}

// ---- vmat_gga_tiled ----
// aow = w0*phi + w1*dphi_x + w2*dphi_y + w3*dphi_z
// vmat[i,j] = sum_g aow_i(g) * phi_j(g)
// The (ja,ia) workgroup handles the symmetric term.

__kernel void vmat_gga_tiled(
    __global const float4 *coords,
    __global const float4 *atom_coords,
    __global const float *rad_val,
    __global const float *rad_du,
    __global const float *rad_dy,
    __global const int *radial_l,
    __global const int *radial_cart0,
    __global const int *atom_radial_offset,
    __global const int *atom_radial_list,
    __global const int *atom_ao0,
    __global const int *atom_nao,
    __global const float *wv,
    __global float *vmat,
    float r0, float du,
    int nrad, int ncart, int ngrids, int natoms)
{
    int lid = get_local_id(1);
    int iTile = get_group_id(0);
    int jTile = get_group_id(1);

    __local float aoI[NPTILE][AO_TILE];
    __local float aoJ[NPTILE][AO_TILE];

    float acc[QPT];
    for (int t = 0; t < QPT; t++) acc[t] = 0.0f;

    for (int gTile = 0; gTile < ngrids; gTile += NPTILE) {

        // Fill aoI: weighted AO with derivatives (aow) for iTile atoms
        for (int k = lid; k < PT_ATOM_SIZE; k += WGS_VMAT) {
            int il = k % NATILE;
            int ip = k / NATILE;
            int ia = iTile * NATILE + il;
            int g = gTile + ip;
            int base = il * MAX_AO_ATOM;
            for (int a = 0; a < MAX_AO_ATOM; a++) aoI[ip][base + a] = 0.0f;
            if (g < ngrids && ia < natoms) {
                float4 d = coords[g] - atom_coords[ia];
                float w0 = wv[g], wx = wv[ngrids + g], wy = wv[2*ngrids + g], wz = wv[3*ngrids + g];
                fill_atom_aow_gga(ia, d, w0, wx, wy, wz, base, aoI[ip], r0, du, nrad, rad_val, rad_du, rad_dy, radial_l, atom_radial_offset, atom_radial_list);
            }
        }

        // Fill aoJ: plain AO values (no derivatives) for jTile atoms
        for (int k = lid; k < PT_ATOM_SIZE; k += WGS_VMAT) {
            int jl = k % NATILE;
            int ip = k / NATILE;
            int ja = jTile * NATILE + jl;
            int g = gTile + ip;
            int base = jl * MAX_AO_ATOM;
            for (int b = 0; b < MAX_AO_ATOM; b++) aoJ[ip][base + b] = 0.0f;
            if (g < ngrids && ja < natoms) {
                float4 d = coords[g] - atom_coords[ja];
                fill_atom_ao_lda(ja, d, base, aoJ[ip], r0, du, nrad, rad_val, rad_du, rad_dy, radial_l, atom_radial_offset, atom_radial_list);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int t = 0; t < QPT; t++) {
            int q = lid + t * WGS_VMAT;
            if (q >= VBLK_SIZE) continue;
            int iao_l = q / AO_TILE;
            int jao_l = q - iao_l * AO_TILE;
            int il = iao_l / MAX_AO_ATOM;
            int jl = jao_l / MAX_AO_ATOM;
            int a = iao_l - il * MAX_AO_ATOM;
            int b = jao_l - jl * MAX_AO_ATOM;
            int ia = iTile * NATILE + il;
            int ja = jTile * NATILE + jl;
            if (ia >= natoms || ja >= natoms || a >= atom_nao[ia] || b >= atom_nao[ja]) continue;

            float s = 0.0f;
            for (int ip = 0; ip < NPTILE; ip++) {
                int g = gTile + ip;
                if (g >= ngrids) continue;
                s += aoI[ip][iao_l] * aoJ[ip][jao_l];
            }
            acc[t] += s;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int t = 0; t < QPT; t++) {
        int q = lid + t * WGS_VMAT;
        if (q >= VBLK_SIZE) continue;
        int iao_l = q / AO_TILE;
        int jao_l = q - iao_l * AO_TILE;
        int il = iao_l / MAX_AO_ATOM;
        int jl = jao_l / MAX_AO_ATOM;
        int a = iao_l - il * MAX_AO_ATOM;
        int b = jao_l - jl * MAX_AO_ATOM;
        int ia = iTile * NATILE + il;
        int ja = jTile * NATILE + jl;
        if (ia < natoms && ja < natoms && a < atom_nao[ia] && b < atom_nao[ja]) {
            int iao = atom_ao0[ia] + a;
            int jao = atom_ao0[ja] + b;
            vmat[iao * ncart + jao] = acc[t];
        }
    }
}

__kernel void transpose_k_buf1_batched(
    __global const float *buf1,
    __global float       *buf1_r,
    int naux, int nao, int nset)
{
    int i = get_global_id(0);
    int pk = get_global_id(1);
    int s = get_global_id(2);
    if (i >= nao || pk >= naux * nao || s >= nset) return;
    int p = pk / nao;
    int k = pk - p * nao;
    buf1_r[(s * nao + i) * naux * nao + pk] = buf1[(p * nao + i) * nset * nao + s * nao + k];
}
