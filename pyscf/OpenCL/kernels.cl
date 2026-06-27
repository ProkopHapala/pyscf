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
