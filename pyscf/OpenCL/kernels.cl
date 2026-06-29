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

// libcint CINTcommon_fac_sp(l), l = 0..7
__constant float CINT_FAC_SP[8] = {
    1.0f, 1.0f, 3.0f, 15.0f, 105.0f, 945.0f, 10395.0f, 135135.0f
};

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

        float fac = CINT_FAC_SP[l];

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

        float fac = CINT_FAC_SP[l];

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
    __global const float *A,   // [M, K] row-major (or rows [row0:row0+M] of larger matrix)
    __global const float *B,   // [K, N] row-major
    __global float       *C,   // [M, N] row-major
    int M, int N, int K, int row0)
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
            Asub[tx][ty] = A[(row0 + row) * K + t * TILE + ty];
        else
            Asub[tx][ty] = 0.0f;

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
// Kernel: matmul_tiled_transpose_A
// C = A^T * B  where A[K,M], B[K,N], C[M,N] all row-major
// A is transposed: A^T[M,K]
// ============================================================
__kernel void matmul_tiled_transpose_A(
    __global const float *A,   // [K, M] row-major rows [a_row0:a_row0+K] of [?, M]
    __global const float *B,   // [K, N] row-major rows [b_row0:b_row0+K] of [?, N]
    __global float       *C,   // [M, N] row-major
    int M, int N, int K, int a_row0, int b_row0)
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
            Asub[tx][ty] = A[(a_row0 + t * TILE + ty) * M + row];
        else
            Asub[tx][ty] = 0.0f;

        if (t * TILE + tx < K && col < N)
            Bsub[tx][ty] = B[(b_row0 + t * TILE + tx) * N + col];
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

__kernel void matmul_tiled_transpose_A_accum(
    __global const float *A,
    __global const float *B,
    __global float       *C,
    int M, int N, int K, int a_row0, int b_row0)
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
            Asub[tx][ty] = A[(a_row0 + t * TILE + ty) * M + row];
        else
            Asub[tx][ty] = 0.0f;
        if (t * TILE + tx < K && col < N)
            Bsub[tx][ty] = B[(b_row0 + t * TILE + tx) * N + col];
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

// PBE XC (libxc maple2c) — see pbe.cl, appended at build time in __init__.py

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
// Precomputed GTO AO on grid — one launch per stage (no Python block loop)
// AO buffers: ao0..ao3 each [ngrids, nao] row-major (same layout as setup_precomputed_gto)
// rho: [4, ngrids] as rho[c*ngrids + g] — matches contract_rho_gga_from_aodm (no extra 2x)
// ============================================================
__kernel void contract_rho_gga_precomp(
    __global const float *ao0,
    __global const float *ao1,
    __global const float *ao2,
    __global const float *ao3,
    __global const float *dm,
    __global float       *rho,
    int nao, int ngrids)
{
    int g = get_global_id(0);
    if (g >= ngrids) return;

    int base = g * nao;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (int i = 0; i < nao; i++) {
        float v0 = ao0[base + i];
        float v1 = ao1[base + i];
        float v2 = ao2[base + i];
        float v3 = ao3[base + i];
        float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
        for (int j = 0; j < nao; j++) {
            float dm_ij = dm[i * nao + j];
            d0 += dm_ij * ao0[base + j];
            d1 += dm_ij * ao1[base + j];
            d2 += dm_ij * ao2[base + j];
            d3 += dm_ij * ao3[base + j];
        }
        s0 += d0 * v0;
        s1 += d0 * v1 + d1 * v0;
        s2 += d0 * v2 + d2 * v0;
        s3 += d0 * v3 + d3 * v0;
    }
    rho[g] = s0;
    rho[ngrids + g] = s1;
    rho[2 * ngrids + g] = s2;
    rho[3 * ngrids + g] = s3;
}

__kernel void contract_rho_lda_precomp(
    __global const float *ao0,
    __global const float *dm,
    __global float       *rho,
    int nao, int ngrids)
{
    int g = get_global_id(0);
    if (g >= ngrids) return;

    int base = g * nao;
    float s0 = 0.0f;
    for (int i = 0; i < nao; i++) {
        float d0 = 0.0f;
        for (int j = 0; j < nao; j++) {
            d0 += dm[i * nao + j] * ao0[base + j];
        }
        s0 += d0 * ao0[base + i];
    }
    rho[g] = s0;
}

__kernel void vxc_mat_gga_precomp(
    __global const float *ao0,
    __global const float *ao1,
    __global const float *ao2,
    __global const float *ao3,
    __global const float *wv,
    __global float       *vmat,
    int nao, int ngrids)
{
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int row = get_group_id(0) * TILE + tx;
    int col = get_group_id(1) * TILE + ty;

    __local float aow_sub[TILE][TILE];
    __local float ao0_sub[TILE][TILE];

    float sum = 0.0f;
    int numGridTiles = (ngrids + TILE - 1) / TILE;

    for (int t = 0; t < numGridTiles; t++) {
        int g_idx = t * TILE + ty;
        if (g_idx < ngrids && row < nao) {
            int base = g_idx * nao + row;
            float val = ao0[base] * wv[g_idx] + ao1[base] * wv[ngrids + g_idx]
                      + ao2[base] * wv[2 * ngrids + g_idx] + ao3[base] * wv[3 * ngrids + g_idx];
            aow_sub[tx][ty] = val;
        } else {
            aow_sub[tx][ty] = 0.0f;
        }

        if (g_idx < ngrids && col < nao) {
            ao0_sub[tx][ty] = ao0[g_idx * nao + col];
        } else {
            ao0_sub[tx][ty] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < TILE; i++) {
            sum += aow_sub[tx][i] * ao0_sub[i][ty];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < nao && col < nao) {
        vmat[row * nao + col] = sum;
    }
}

__kernel void vxc_mat_lda_precomp(
    __global const float *ao0,
    __global const float *wv,
    __global float       *vmat,
    int nao, int ngrids)
{
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int row = get_group_id(0) * TILE + tx;
    int col = get_group_id(1) * TILE + ty;

    __local float aow_sub[TILE][TILE];
    __local float ao0_sub[TILE][TILE];

    float sum = 0.0f;
    int numGridTiles = (ngrids + TILE - 1) / TILE;

    for (int t = 0; t < numGridTiles; t++) {
        int g_idx = t * TILE + ty;
        if (g_idx < ngrids && row < nao) {
            aow_sub[tx][ty] = ao0[g_idx * nao + row] * wv[g_idx];
        } else {
            aow_sub[tx][ty] = 0.0f;
        }

        if (g_idx < ngrids && col < nao) {
            ao0_sub[tx][ty] = ao0[g_idx * nao + col];
        } else {
            ao0_sub[tx][ty] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

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
    int nao, int nblk, int grid0, int ao_row0)
{
    int g = get_global_id(0);
    if (g >= nblk) return;

    float s0 = 0.0f;
    int base_ao = (ao_row0 + g) * nao;
    int base_dm = g * nao;
    for (int i = 0; i < nao; i++) {
        s0 += aodm0[base_dm + i] * ao0[base_ao + i];
    }
    rho[grid0 + g] = s0;
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
    int nao, int nblk, int grid0, int ngrids_total, int ao_row0)
{
    int g = get_global_id(0);
    if (g >= nblk) return;

    float s0 = 0.0f;
    float s1 = 0.0f;
    float s2 = 0.0f;
    float s3 = 0.0f;
    int base_ao = (ao_row0 + g) * nao;
    int base_dm = g * nao;
    for (int i = 0; i < nao; i++) {
        float v0 = ao0[base_ao + i];
        float v1 = ao1[base_ao + i];
        float v2 = ao2[base_ao + i];
        float v3 = ao3[base_ao + i];
        float d0 = aodm0[base_dm + i];
        s0 += d0 * v0;
        s1 += d0 * v1 + aodm1[base_dm + i] * v0;
        s2 += d0 * v2 + aodm2[base_dm + i] * v0;
        s3 += d0 * v3 + aodm3[base_dm + i] * v0;
    }
    int og = grid0 + g;
    rho[og] = s0;
    rho[ngrids_total + og] = s1;
    rho[2*ngrids_total + og] = s2;
    rho[3*ngrids_total + og] = s3;
}

__kernel void scale_aow_lda(
    __global const float *ao0,
    __global const float *wv,
    __global float       *aow,
    int nao, int nblk, int grid0, int ao_row0)
{
    int g = get_global_id(0);
    int i = get_global_id(1);
    if (g >= nblk || i >= nao) return;

    int idx = g * nao + i;
    aow[idx] = ao0[(ao_row0 + g) * nao + i] * wv[grid0 + g];
}

__kernel void scale_aow_gga_split(
    __global const float *ao0,
    __global const float *ao1,
    __global const float *ao2,
    __global const float *ao3,
    __global const float *wv,
    __global float       *aow,
    int nao, int nblk, int grid0, int ngrids_total, int ao_row0)
{
    int g = get_global_id(0);
    int i = get_global_id(1);
    if (g >= nblk || i >= nao) return;

    int idx = g * nao + i;
    int og = grid0 + g;
    int aidx = (ao_row0 + g) * nao + i;
    aow[idx] = ao0[aidx] * wv[og] + ao1[aidx] * wv[ngrids_total + og] + ao2[aidx] * wv[2*ngrids_total + og] + ao3[aidx] * wv[3*ngrids_total + og];
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

// float2 knot: (y, dy/du). Interval [ik, ik+1] uses rad_node[ik], rad_node[ik+1].
inline void hermite_map_point(float4 d, float r0, float du, int nrad, float *t, float *t1m, int *ik)
{
    float r = sqrt(d.x*d.x + d.y*d.y + d.z*d.z);
    float uf = log1p(r / r0) / du;
    int i = max(0, min((int)floor(uf), nrad - 2));
    *ik = i;
    *t = clamp(uf - (float)i, 0.0f, 1.0f);
    *t1m = *t - 1.0f;
}

inline float hermite_eval_node(float t, float t1m, float du, float2 n0, float2 n1)
{
    float y0 = n0.x, d0 = n0.y, y1 = n1.x, d1 = n1.y;
    float dy = y1 - y0;
    return y0 + t*t*(3.0f-2.0f*t)*dy + t*t1m*t1m*du*d0 + t*t*t1m*du*d1;
}

inline float hermite_eval_deriv_node(float t, float t1m, float du, float2 n0, float2 n1)
{
    float dy = n1.x - n0.x;
    float d0 = n0.y, d1 = n1.y;
    return (6.0f*t*(1.0f-t)*dy + (3.0f*t-1.0f)*t1m*du*d0 + t*(3.0f*t-2.0f)*du*d1) / du;
}

inline float hermite_eval_ir(float t, float t1m, float du, int ik, int ir, int nrad, __global const float2 *rad_node)
{
    int base = ir * nrad + ik;
    return hermite_eval_node(t, t1m, du, rad_node[base], rad_node[base + 1]);
}

inline float hermite_eval_deriv_ir(float t, float t1m, float du, int ik, int ir, int nrad, __global const float2 *rad_node)
{
    int base = ir * nrad + ik;
    return hermite_eval_deriv_node(t, t1m, du, rad_node[base], rad_node[base + 1]);
}

__kernel void eval_ao_mapped_hermite_cart(
    __global const float *coords,
    __global const float *atom_coords,
    __global const float2 *rad_node,
    __global const int   *shell_ctr_ir,
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
    float t1m = t - 1.0f;
    int ir = shell_ctr_ir[sh * nctr_max + ctr];
    float radial = hermite_eval_ir(t, t1m, du, i, ir, nrad, rad_node);
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
    __global const float2 *rad_node,
    __global const int   *shell_ctr_ir,
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
    float t1m = t - 1.0f;
    int ir = shell_ctr_ir[sh * nctr_max + ctr];
    float radial = hermite_eval_ir(t, t1m, du, i, ir, nrad, rad_node);
    float drad_du = hermite_eval_deriv_ir(t, t1m, du, i, ir, nrad, rad_node);
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
    __global const float2 *rad_node,
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
        float radial = hermite_eval_ir(t, t1m, du, i, ir, nrad, rad_node);
        eval_radial_cart(d, radial_l[ir], g * ncart + radial_cart0[ir], radial, ncart, ao);
    }
}

__kernel void eval_ao_mapped_hermite_cart_deriv1_atom(
    __global const float4 *coords,
    __global const float4 *atom_coords,
    __global const float2 *rad_node,
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
        float radial = hermite_eval_ir(t, t1m, du, i, ir, nrad, rad_node);
        float drad_du = hermite_eval_deriv_ir(t, t1m, du, i, ir, nrad, rad_node);
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
// - rho: one workgroup per grid tile; loop all iTile x jTile inside;
//   reduce over NATILE in __local psum → final rho[g] (no partials, no atomics)
// ============================================================

#ifndef NPTILE
#define NPTILE 16
#endif
#ifndef LOG_NPTILE
#define LOG_NPTILE 4
#endif
#ifndef NATILE
#define NATILE 4
#endif
#ifndef LOG_NATILE
#define LOG_NATILE 2
#endif
#define PT_ATOM_SIZE (NPTILE * NATILE)
#ifndef WGS_TILED
#define WGS_TILED (NPTILE * NATILE)
#endif
#ifndef MAX_SHELL
#define MAX_SHELL 6
#endif
#ifndef MAX_AO_ATOM
#define MAX_AO_ATOM 16
#endif
#ifndef LOG_MAX_AO_ATOM
#define LOG_MAX_AO_ATOM 4
#endif
#ifndef MAX_ITILE
#define MAX_ITILE 32
#endif
#define AO_BLK (MAX_AO_ATOM * MAX_AO_ATOM)
#define LOG_AO_BLK (LOG_MAX_AO_ATOM + LOG_MAX_AO_ATOM)
#define AO_TILE (NATILE * MAX_AO_ATOM)
#define LOG_AO_TILE (LOG_NATILE + LOG_MAX_AO_ATOM)
#define DM_TILE_SIZE (NATILE * NATILE * AO_BLK)
#define WFJ_SIZE (NPTILE * NATILE * MAX_SHELL)

inline void decode_q_vmat(int q, int *iao_l, int *jao_l, int *il, int *jl, int *a, int *b)
{
    int jao = q & (AO_TILE - 1);
    int iao = q >> LOG_AO_TILE;
    *jao_l = jao;
    *iao_l = iao;
    *il = iao >> LOG_MAX_AO_ATOM;
    *jl = jao >> LOG_MAX_AO_ATOM;
    *a = iao & (MAX_AO_ATOM - 1);
    *b = jao & (MAX_AO_ATOM - 1);
}

// Evaluate all radial channels for one atom at one point (ir_list in private or local).
inline void eval_radials_slice(int ns, const int *ir_list, float t, float t1m, float du, int ik, int nrad, __global const float2 *rad_node, float *Ri)
{
    for (int s = 0; s < ns; s++)
        Ri[s] = hermite_eval_ir(t, t1m, du, ik, ir_list[s], nrad, rad_node);
    for (int s = ns; s < MAX_SHELL; s++) Ri[s] = 0.0f;
}

inline void eval_radials_slice_deriv(int ns, const int *ir_list, float t, float t1m, float du, int ik, int nrad, float r, float r0, __global const float2 *rad_node, float *Ri, float *dRi)
{
    for (int s = 0; s < ns; s++) {
        Ri[s] = hermite_eval_ir(t, t1m, du, ik, ir_list[s], nrad, rad_node);
        dRi[s] = hermite_eval_deriv_ir(t, t1m, du, ik, ir_list[s], nrad, rad_node) / (r + r0);
    }
    for (int s = ns; s < MAX_SHELL; s++) { Ri[s] = 0.0f; dRi[s] = 0.0f; }
}

inline void load_atom_tile_meta(int tile0, int natoms, int lid, int lstride, __global const int *atom_radial_offset, __global const int *atom_radial_list, __local int l_ns[NATILE], __local int l_ir[NATILE][MAX_SHELL])
{
    for (int k = lid; k < NATILE * MAX_SHELL; k += lstride) {
        int jj = k / MAX_SHELL;
        int s = k % MAX_SHELL;
        int ja = tile0 + jj;
        if (ja < natoms) {
            int off = atom_radial_offset[ja];
            int ns = atom_radial_offset[ja + 1] - off;
            if (s == 0) l_ns[jj] = ns;
            if (s < ns) l_ir[jj][s] = atom_radial_list[off + s];
        } else if (s == 0) {
            l_ns[jj] = 0;
        }
    }
}

inline void load_atom_tile_meta_l(int tile0, int natoms, int lid, int lstride, __global const int *atom_radial_offset, __global const int *atom_radial_list, __global const int *radial_l, __local int l_ns[NATILE], __local int l_ir[NATILE][MAX_SHELL], __local int l_l[NATILE][MAX_SHELL])
{
    for (int k = lid; k < NATILE * MAX_SHELL; k += lstride) {
        int jj = k / MAX_SHELL;
        int s = k % MAX_SHELL;
        int ja = tile0 + jj;
        if (ja < natoms) {
            int off = atom_radial_offset[ja];
            int ns = atom_radial_offset[ja + 1] - off;
            if (s == 0) l_ns[jj] = ns;
            if (s < ns) {
                int ir = atom_radial_list[off + s];
                l_ir[jj][s] = ir;
                l_l[jj][s] = radial_l[ir];
            }
        } else if (s == 0) {
            l_ns[jj] = 0;
        }
    }
}

inline int load_atom_ir(int ia, int natoms, __global const int *atom_radial_offset, __global const int *atom_radial_list, int *ir_out)
{
    if (ia >= natoms) return 0;
    int off = atom_radial_offset[ia];
    int ns = atom_radial_offset[ia + 1] - off;
    for (int s = 0; s < ns; s++) ir_out[s] = atom_radial_list[off + s];
    return ns;
}

inline int load_atom_ir_l(int ia, int natoms, __global const int *atom_radial_offset, __global const int *atom_radial_list, __global const int *radial_l, int *ir_out, int *l_out)
{
    if (ia >= natoms) return 0;
    int off = atom_radial_offset[ia];
    int ns = atom_radial_offset[ia + 1] - off;
    for (int s = 0; s < ns; s++) {
        int ir = atom_radial_list[off + s];
        ir_out[s] = ir;
        l_out[s] = radial_l[ir];
    }
    return ns;
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
        f0[0]=R*x; f1[0]=gx*x+R; f2[0]=gy*x;   f3[0]=gz*x;
        f0[1]=R*y; f1[1]=gx*y;   f2[1]=gy*y+R; f3[1]=gz*y;
        f0[2]=R*z; f1[2]=gx*z;   f2[2]=gy*z;   f3[2]=gz*z+R;
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

// float4 per cart component: (phi, dphi/dx, dphi/dy, dphi/dz)
inline int unfold_shell_deriv_f4(int l, float R, float dRdr, float4 d, float invr, float4 *f)
{
    float x=d.x, y=d.y, z=d.z;
    float gx=dRdr*x*invr, gy=dRdr*y*invr, gz=dRdr*z*invr;
    if (l == 0) { f[0]=(float4)(R, gx, gy, gz); return 1; }
    if (l == 1) {
        f[0]=(float4)(R*x, gx*x+R, gy*x, gz*x);
        f[1]=(float4)(R*y, gx*y,   gy*y+R, gz*y);
        f[2]=(float4)(R*z, gx*z,   gy*z,   gz*z+R);
        return 3;
    }
    float x2=x*x, y2=y*y, z2=z*z, xy=x*y, xz=x*z, yz=y*z;
    f[0]=(float4)(R*x2, gx*x2+2.f*R*x, gy*x2,            gz*x2);
    f[1]=(float4)(R*xy, gx*xy+R*y,     gy*xy+R*x,        gz*xy);
    f[2]=(float4)(R*xz, gx*xz+R*z,     gy*xz,            gz*xz+R*x);
    f[3]=(float4)(R*y2, gx*y2,         gy*y2+2.f*R*y,    gz*y2);
    f[4]=(float4)(R*yz, gx*yz,         gy*yz+R*z,        gz*yz+R*y);
    f[5]=(float4)(R*z2, gx*z2,         gy*z2,            gz*z2+2.f*R*z);
    return 6;
}

inline void accum_rho_gga_f4(float4 fi, float4 t, float *rho_val, float *gx_val, float *gy_val, float *gz_val)
{
    *rho_val += fi.x * t.x;
    *gx_val  += fi.x * t.y + fi.y * t.x;
    *gy_val  += fi.x * t.z + fi.z * t.x;
    *gz_val  += fi.x * t.w + fi.w * t.x;
}

// Contract one (ia,ja) atom pair using shellwise unfolding.
// Uses private Ri, local Rj, local dm_blk. No full phi arrays.
inline float contract_pair_rho(float4 di, float4 dj,
    int il, int jl, float *Ri, __local float *Rj,
    __local float dm_blk[NATILE][NATILE][MAX_AO_ATOM][MAX_AO_ATOM],
    int ns_i, const int *ir_i, int ns_j, const int *ir_j,
    __global const int *radial_l) {
    float acc = 0.0f;
    int iao_off = 0;
    for (int si = 0; si < ns_i; si++) {
        int li = radial_l[ir_i[si]];
        float fi[6]; int ni = unfold_shell(li, Ri[si], di, fi);
        int jao_off = 0;
        for (int sj = 0; sj < ns_j; sj++) {
            int lj = radial_l[ir_j[sj]];
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

// v2: angular momentum l[] preloaded; no radial_l global reads in contract
inline float contract_pair_rho_v2(int il, int jl, float *Ri, __local float *Rj,
    __local float dm_blk[NATILE][NATILE][MAX_AO_ATOM][MAX_AO_ATOM],
    int ns_i, const int *l_i, int ns_j, const int *l_j,
    float4 di, float4 dj) {
    float acc = 0.0f;
    int iao_off = 0;
    for (int si = 0; si < ns_i; si++) {
        float fi[6]; int ni = unfold_shell(l_i[si], Ri[si], di, fi);
        int jao_off = 0;
        for (int sj = 0; sj < ns_j; sj++) {
            float fj[6]; int nj = unfold_shell(l_j[sj], Rj[sj], dj, fj);
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
inline void contract_pair_rho_gga(float4 di, float4 dj,
    int il, int jl, float *Ri, float *dRi, __local float *Rj, __local float *dRj,
    __local float dm_blk[NATILE][NATILE][MAX_AO_ATOM][MAX_AO_ATOM],
    int ns_i, const int *ir_i, int ns_j, const int *ir_j,
    __global const int *radial_l,
    float *rho_val, float *gx_val, float *gy_val, float *gz_val) {
    *rho_val = 0.0f; *gx_val = 0.0f; *gy_val = 0.0f; *gz_val = 0.0f;
    float ri_mag = sqrt(di.x*di.x + di.y*di.y + di.z*di.z);
    float rj_mag = sqrt(dj.x*dj.x + dj.y*dj.y + dj.z*dj.z);
    float invr_i = ri_mag > 1e-20f ? 1.0f/ri_mag : 0.0f;
    float invr_j = rj_mag > 1e-20f ? 1.0f/rj_mag : 0.0f;
    int iao_off = 0;
    for (int si = 0; si < ns_i; si++) {
        int li = radial_l[ir_i[si]];
        float fi0[6], fi1[6], fi2[6], fi3[6];
        int ni = unfold_shell_deriv(li, Ri[si], dRi[si], di, invr_i, fi0, fi1, fi2, fi3);
        int jao_off = 0;
        for (int sj = 0; sj < ns_j; sj++) {
            int lj = radial_l[ir_j[sj]];
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
                *rho_val +=                fi0[ai] * t0;
                *gx_val  += fi0[ai] * t1 + fi1[ai] * t0;
                *gy_val  += fi0[ai] * t2 + fi2[ai] * t0;
                *gz_val  += fi0[ai] * t3 + fi3[ai] * t0;
            }
            jao_off += nj;
        }
        iao_off += ni;
    }
}

// v2: l[] preloaded, invr precomputed, float4 (phi,dphi) inner dm·fj
inline void contract_pair_rho_gga_v2(int il, int jl, float *Ri, float *dRi, __local float *Rj, __local float *dRj,
    __local float dm_blk[NATILE][NATILE][MAX_AO_ATOM][MAX_AO_ATOM],
    int ns_i, const int *l_i, int ns_j, const int *l_j,
    float4 di, float4 dj, float invr_i, float invr_j,
    float *rho_val, float *gx_val, float *gy_val, float *gz_val) {
    *rho_val = 0.0f; *gx_val = 0.0f; *gy_val = 0.0f; *gz_val = 0.0f;
    int iao_off = 0;
    for (int si = 0; si < ns_i; si++) {
        float4 fi[6];
        int ni = unfold_shell_deriv_f4(l_i[si], Ri[si], dRi[si], di, invr_i, fi);
        int jao_off = 0;
        for (int sj = 0; sj < ns_j; sj++) {
            float4 fj[6];
            int nj = unfold_shell_deriv_f4(l_j[sj], Rj[sj], dRj[sj], dj, invr_j, fj);
            for (int ai = 0; ai < ni; ai++) {
                float4 t = (float4)(0.0f);
                for (int aj = 0; aj < nj; aj++) {
                    float dm = dm_blk[il][jl][iao_off+ai][jao_off+aj];
                    t += dm * fj[aj];
                }
                accum_rho_gga_f4(fi[ai], t, rho_val, gx_val, gy_val, gz_val);
            }
            jao_off += nj;
        }
        iao_off += ni;
    }
}

// ---- rho_lda_tiled ----
// One workgroup per grid-point tile (NPTILE points).
// Threads (ip, il): loop all iTile x jTile atom-pair tiles inside the kernel.
// Final __local reduction over il → rho[g].

__kernel void rho_lda_tiled(
    __global const float4 *coords,
    __global const float4 *atom_coords,
    __global const float2 *rad_node,
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
    int g = gTile * NPTILE + ip;

    __local float wfRj[NPTILE][NATILE][MAX_SHELL];           // j-atom radial values at each grid point in tile (cooperative fill)
    __local float dm_blk[NATILE][NATILE][MAX_AO_ATOM][MAX_AO_ATOM]; // DM sub-block for current (iTile,jTile) pair
    __local float psum[WGS_TILED];                           // per-thread rho partials before NATILE reduction
    __local int l_j_ns[NATILE];                              // shell count per j-atom in current jTile
    __local int l_j_ir[NATILE][MAX_SHELL];                   // radial channel index ir per (j-atom, shell)
    __local int l_j_l[NATILE][MAX_SHELL];                     // angular momentum l per (j-atom, shell)

    float rho_priv = 0.0f;
    int n_iTiles = (natoms + NATILE - 1) / NATILE;
    int i_ir_tile[MAX_ITILE][MAX_SHELL];
    int i_l_tile[MAX_ITILE][MAX_SHELL];
    int i_ns_tile[MAX_ITILE];
    float Ri_tile[MAX_ITILE][MAX_SHELL];
    float4 di_tile[MAX_ITILE];

    for (int it = 0; it < n_iTiles; it++) {
        int ia = (it << LOG_NATILE) + il;
        i_ns_tile[it] = load_atom_ir_l(ia, natoms, atom_radial_offset, atom_radial_list, radial_l, i_ir_tile[it], i_l_tile[it]);
        di_tile[it] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        float t_i = 0.0f, t1m_i = 0.0f; int ik_i = 0;
        for (int s = 0; s < MAX_SHELL; s++) Ri_tile[it][s] = 0.0f;
        if (g < ngrids && ia < natoms) {
            di_tile[it] = coords[g] - atom_coords[ia];
            hermite_map_point(di_tile[it], r0, du, nrad, &t_i, &t1m_i, &ik_i);
            eval_radials_slice(i_ns_tile[it], i_ir_tile[it], t_i, t1m_i, du, ik_i, nrad, rad_node, Ri_tile[it]);
        }
    }

    for (int jTile = 0; jTile < natoms; jTile += NATILE) {
        load_atom_tile_meta_l(jTile, natoms, lid, WGS_TILED, atom_radial_offset, atom_radial_list, radial_l, l_j_ns, l_j_ir, l_j_l);
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = lid; k < PT_ATOM_SIZE; k += WGS_TILED) {
            int jj = k & (NATILE - 1);
            int pp = k >> LOG_NATILE;
            int gj = gTile * NPTILE + pp;
            int ja = jTile + jj;
            for (int s = 0; s < MAX_SHELL; s++) wfRj[pp][jj][s] = 0.0f;
            if (gj < ngrids && ja < natoms) {
                float t, t1m; int ik;
                hermite_map_point(coords[gj] - atom_coords[ja], r0, du, nrad, &t, &t1m, &ik);
                for (int s = 0; s < l_j_ns[jj]; s++)
                    wfRj[pp][jj][s] = hermite_eval_ir(t, t1m, du, ik, l_j_ir[jj][s], nrad, rad_node);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int it = 0; it < n_iTiles; it++) {
            int ia = (it << LOG_NATILE) + il;

            for (int k = lid; k < DM_TILE_SIZE; k += WGS_TILED) {
                int ab = k & (AO_BLK - 1);
                int pq = k >> LOG_AO_BLK;
                int a = ab >> LOG_MAX_AO_ATOM;
                int b = ab & (MAX_AO_ATOM - 1);
                int ii2 = pq >> LOG_NATILE;
                int jj2 = pq & (NATILE - 1);
                int ia2 = (it << LOG_NATILE) + ii2, ja2 = jTile + jj2;
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
                    rho_priv += contract_pair_rho_v2(il, jl, Ri_tile[it], wfRj[ip][jl], dm_blk, i_ns_tile[it], i_l_tile[it], l_j_ns[jl], &l_j_l[jl][0], di_tile[it], dj);
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    psum[lid] = rho_priv;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (il == 0 && g < ngrids) {
        float s = 0.0f;
        for (int k = 0; k < NATILE; k++) s += psum[k * NPTILE + ip];
        rho[g] = s;
    }
}

// ---- rho_gga_tiled ----
// Same structure as rho_lda_tiled; also accumulates grad rho components.

__kernel void rho_gga_tiled(
    __global const float4 *coords,
    __global const float4 *atom_coords,
    __global const float2 *rad_node,
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
    int g = gTile * NPTILE + ip;

    __local float wfRj[NPTILE][NATILE][MAX_SHELL];           // j-atom radial R(r) at each grid point in tile
    __local float dwfRj[NPTILE][NATILE][MAX_SHELL];          // j-atom dR/dr at each grid point in tile
    __local float dm_blk[NATILE][NATILE][MAX_AO_ATOM][MAX_AO_ATOM]; // DM sub-block for current (iTile,jTile)
    __local float psum_rho[WGS_TILED];                       // rho partial per thread before NATILE reduction
    __local float psum_gx[WGS_TILED];                        // grad_x partial per thread
    __local float psum_gy[WGS_TILED];                        // grad_y partial per thread
    __local float psum_gz[WGS_TILED];                        // grad_z partial per thread
    __local int l_j_ns[NATILE];                              // shell count per j-atom in current jTile
    __local int l_j_ir[NATILE][MAX_SHELL];                   // radial channel ir per (j-atom, shell)
    __local int l_j_l[NATILE][MAX_SHELL];                     // angular momentum l per (j-atom, shell)

    float rho_priv = 0.0f, gx_priv = 0.0f, gy_priv = 0.0f, gz_priv = 0.0f;
    int n_iTiles = (natoms + NATILE - 1) / NATILE;
    int i_ir_tile[MAX_ITILE][MAX_SHELL];
    int i_l_tile[MAX_ITILE][MAX_SHELL];
    int i_ns_tile[MAX_ITILE];
    float Ri_tile[MAX_ITILE][MAX_SHELL];
    float dRi_tile[MAX_ITILE][MAX_SHELL];
    float4 di_tile[MAX_ITILE];
    float invr_i_tile[MAX_ITILE];

    for (int it = 0; it < n_iTiles; it++) {
        int ia = (it << LOG_NATILE) + il;
        i_ns_tile[it] = load_atom_ir_l(ia, natoms, atom_radial_offset, atom_radial_list, radial_l, i_ir_tile[it], i_l_tile[it]);
        di_tile[it] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        invr_i_tile[it] = 0.0f;
        float t_i = 0.0f, t1m_i = 0.0f; int ik_i = 0;
        float r_i = 0.0f;
        for (int s = 0; s < MAX_SHELL; s++) { Ri_tile[it][s] = 0.0f; dRi_tile[it][s] = 0.0f; }
        if (g < ngrids && ia < natoms) {
            di_tile[it] = coords[g] - atom_coords[ia];
            r_i = sqrt(di_tile[it].x*di_tile[it].x + di_tile[it].y*di_tile[it].y + di_tile[it].z*di_tile[it].z);
            invr_i_tile[it] = r_i > 1e-20f ? 1.0f/r_i : 0.0f;
            hermite_map_point(di_tile[it], r0, du, nrad, &t_i, &t1m_i, &ik_i);
            eval_radials_slice_deriv(i_ns_tile[it], i_ir_tile[it], t_i, t1m_i, du, ik_i, nrad, r_i, r0, rad_node, Ri_tile[it], dRi_tile[it]);
        }
    }

    for (int jTile = 0; jTile < natoms; jTile += NATILE) {
        load_atom_tile_meta_l(jTile, natoms, lid, WGS_TILED, atom_radial_offset, atom_radial_list, radial_l, l_j_ns, l_j_ir, l_j_l);
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = lid; k < PT_ATOM_SIZE; k += WGS_TILED) {
            int jj = k & (NATILE - 1);
            int pp = k >> LOG_NATILE;
            int gj = gTile * NPTILE + pp;
            int ja = jTile + jj;
            for (int s = 0; s < MAX_SHELL; s++) { wfRj[pp][jj][s] = 0.0f; dwfRj[pp][jj][s] = 0.0f; }
            if (gj < ngrids && ja < natoms) {
                float4 dj = coords[gj] - atom_coords[ja];
                float rj = sqrt(dj.x*dj.x + dj.y*dj.y + dj.z*dj.z);
                float t, t1m; int ik;
                hermite_map_point(dj, r0, du, nrad, &t, &t1m, &ik);
                for (int s = 0; s < l_j_ns[jj]; s++) {
                    wfRj[pp][jj][s] = hermite_eval_ir(t, t1m, du, ik, l_j_ir[jj][s], nrad, rad_node);
                    dwfRj[pp][jj][s] = hermite_eval_deriv_ir(t, t1m, du, ik, l_j_ir[jj][s], nrad, rad_node) / (rj + r0);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int it = 0; it < n_iTiles; it++) {
            int ia = (it << LOG_NATILE) + il;

            for (int k = lid; k < DM_TILE_SIZE; k += WGS_TILED) {
                int ab = k & (AO_BLK - 1);
                int pq = k >> LOG_AO_BLK;
                int a = ab >> LOG_MAX_AO_ATOM;
                int b = ab & (MAX_AO_ATOM - 1);
                int ii2 = pq >> LOG_NATILE;
                int jj2 = pq & (NATILE - 1);
                int ia2 = (it << LOG_NATILE) + ii2, ja2 = jTile + jj2;
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
                    float rj = sqrt(dj.x*dj.x + dj.y*dj.y + dj.z*dj.z);
                    float invr_j = rj > 1e-20f ? 1.0f/rj : 0.0f;
                    float rv, gv1, gv2, gv3;
                    contract_pair_rho_gga_v2(il, jl, Ri_tile[it], dRi_tile[it], wfRj[ip][jl], dwfRj[ip][jl], dm_blk, i_ns_tile[it], i_l_tile[it], l_j_ns[jl], &l_j_l[jl][0], di_tile[it], dj, invr_i_tile[it], invr_j, &rv, &gv1, &gv2, &gv3);
                    rho_priv += rv; gx_priv += gv1; gy_priv += gv2; gz_priv += gv3;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
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
        rho[g] = sr;
        rho[ngrids + g] = sx;
        rho[2*ngrids + g] = sy;
        rho[3*ngrids + g] = sz;
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
#define VBLK_SIZE (AO_TILE * AO_TILE)
#define QPT ((VBLK_SIZE + WGS_VMAT - 1) / WGS_VMAT)
#define PT_ATOM_SIZE (NPTILE * NATILE)

// Fill local AO values for one atom at one grid point (LDA: just phi)
inline void fill_atom_ao_lda(int ns, const int *ir_list, float4 d, int base, __local float *ao, float r0, float du, int nrad, __global const float2 *rad_node, __global const int *radial_l)
{
    float t, t1m; int ik;
    hermite_map_point(d, r0, du, nrad, &t, &t1m, &ik);
    float R[MAX_SHELL];
    eval_radials_slice(ns, ir_list, t, t1m, du, ik, nrad, rad_node, R);
    int ao0 = 0;
    for (int s = 0; s < ns; s++) {
        float f[6];
        int n = unfold_shell(radial_l[ir_list[s]], R[s], d, f);
        for (int a = 0; a < n; a++) ao[base + ao0 + a] = f[a];
        ao0 += n;
    }
}

inline void fill_atom_aow_gga(int ns, const int *ir_list, float4 d, float w0, float wx, float wy, float wz, int base, __local float *ao, float r0, float du, int nrad, __global const float2 *rad_node, __global const int *radial_l)
{
    float t, t1m; int ik;
    hermite_map_point(d, r0, du, nrad, &t, &t1m, &ik);
    float r = sqrt(d.x*d.x + d.y*d.y + d.z*d.z);
    float R[MAX_SHELL], dR[MAX_SHELL];
    eval_radials_slice_deriv(ns, ir_list, t, t1m, du, ik, nrad, r, r0, rad_node, R, dR);
    float invr = (r > 1e-20f) ? 1.0f / r : 0.0f;
    int ao0 = 0;
    for (int s = 0; s < ns; s++) {
        float f0[6], f1[6], f2[6], f3[6];
        int n = unfold_shell_deriv(radial_l[ir_list[s]], R[s], dR[s], d, invr, f0, f1, f2, f3);
        for (int a = 0; a < n; a++) ao[base + ao0 + a] = w0*f0[a] + wx*f1[a] + wy*f2[a] + wz*f3[a];
        ao0 += n;
    }
}

__kernel void vmat_lda_tiled(
    __global const float4 *coords,
    __global const float4 *atom_coords,
    __global const float2 *rad_node,
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
    if (jTile < iTile) return;

    __local float aoI[NPTILE][AO_TILE];                      // unfolded i-atom Cartesian AO values per grid point
    __local float aoJ[NPTILE][AO_TILE];                      // unfolded j-atom Cartesian AO values per grid point
    __local int l_i_ns[NATILE];                              // shell count per i-atom in this workgroup's iTile
    __local int l_j_ns[NATILE];                              // shell count per j-atom in this workgroup's jTile
    __local int l_i_ir[NATILE][MAX_SHELL];                   // radial channel ir per (i-atom, shell)
    __local int l_j_ir[NATILE][MAX_SHELL];                   // radial channel ir per (j-atom, shell)

    load_atom_tile_meta(iTile << LOG_NATILE, natoms, lid, WGS_VMAT, atom_radial_offset, atom_radial_list, l_i_ns, l_i_ir);
    load_atom_tile_meta(jTile << LOG_NATILE, natoms, lid, WGS_VMAT, atom_radial_offset, atom_radial_list, l_j_ns, l_j_ir);
    barrier(CLK_LOCAL_MEM_FENCE);

    float acc[QPT];
    for (int t = 0; t < QPT; t++) acc[t] = 0.0f;

    for (int gTile = 0; gTile < ngrids; gTile += NPTILE) {

        // Fill aoI: unfolded AO values for iTile atoms
        for (int k = lid; k < PT_ATOM_SIZE; k += WGS_VMAT) {
            int il = k & (NATILE - 1);
            int ip = k >> LOG_NATILE;
            int ia = (iTile << LOG_NATILE) + il;
            int g = gTile + ip;
            int base = il << LOG_MAX_AO_ATOM;
            for (int a = 0; a < MAX_AO_ATOM; a++) aoI[ip][base + a] = 0.0f;
            if (g < ngrids && ia < natoms) {
                float4 d = coords[g] - atom_coords[ia];
                fill_atom_ao_lda(l_i_ns[il], &l_i_ir[il][0], d, base, aoI[ip], r0, du, nrad, rad_node, radial_l);
            }
        }

        // Fill aoJ: unfolded AO values for jTile atoms
        for (int k = lid; k < PT_ATOM_SIZE; k += WGS_VMAT) {
            int jl = k & (NATILE - 1);
            int ip = k >> LOG_NATILE;
            int ja = (jTile << LOG_NATILE) + jl;
            int g = gTile + ip;
            int base = jl << LOG_MAX_AO_ATOM;
            for (int b = 0; b < MAX_AO_ATOM; b++) aoJ[ip][base + b] = 0.0f;
            if (g < ngrids && ja < natoms) {
                float4 d = coords[g] - atom_coords[ja];
                fill_atom_ao_lda(l_j_ns[jl], &l_j_ir[jl][0], d, base, aoJ[ip], r0, du, nrad, rad_node, radial_l);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Each thread accumulates QPT AO-pair elements over NPTILE grid points
        for (int t = 0; t < QPT; t++) {
            int q = lid + t * WGS_VMAT;
            if (q >= VBLK_SIZE) continue;
            int iao_l, jao_l, il, jl, a, b;
            decode_q_vmat(q, &iao_l, &jao_l, &il, &jl, &a, &b);
            int ia = (iTile << LOG_NATILE) + il;
            int ja = (jTile << LOG_NATILE) + jl;
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
        int iao_l, jao_l, il, jl, a, b;
        decode_q_vmat(q, &iao_l, &jao_l, &il, &jl, &a, &b);
        int ia = (iTile << LOG_NATILE) + il;
        int ja = (jTile << LOG_NATILE) + jl;
        if (ia < natoms && ja < natoms && a < atom_nao[ia] && b < atom_nao[ja]) {
            int iao = atom_ao0[ia] + a;
            int jao = atom_ao0[ja] + b;
            vmat[iao * ncart + jao] = acc[t];
            if (iTile != jTile) vmat[jao * ncart + iao] = acc[t];
        }
    }
}

// ---- vmat_gga_tiled ----
// aow = w0*phi + w1*dphi_x + w2*dphi_y + w3*dphi_z
// vmat[i,j] = sum_g aow_i(g) * phi_j(g)
// The host adds vmat + vmat.T for the symmetric GGA operator.

__kernel void vmat_gga_tiled(
    __global const float4 *coords,
    __global const float4 *atom_coords,
    __global const float2 *rad_node,
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

    __local float aoI[NPTILE][AO_TILE];                      // unfolded i-atom Cartesian AO values per grid point
    __local float aoJ[NPTILE][AO_TILE];                      // unfolded j-atom Cartesian AO values per grid point
    __local int l_i_ns[NATILE];                              // shell count per i-atom in this workgroup's iTile
    __local int l_j_ns[NATILE];                              // shell count per j-atom in this workgroup's jTile
    __local int l_i_ir[NATILE][MAX_SHELL];                   // radial channel ir per (i-atom, shell)
    __local int l_j_ir[NATILE][MAX_SHELL];                   // radial channel ir per (j-atom, shell)

    load_atom_tile_meta(iTile << LOG_NATILE, natoms, lid, WGS_VMAT, atom_radial_offset, atom_radial_list, l_i_ns, l_i_ir);
    load_atom_tile_meta(jTile << LOG_NATILE, natoms, lid, WGS_VMAT, atom_radial_offset, atom_radial_list, l_j_ns, l_j_ir);
    barrier(CLK_LOCAL_MEM_FENCE);

    float acc[QPT];
    for (int t = 0; t < QPT; t++) acc[t] = 0.0f;

    for (int gTile = 0; gTile < ngrids; gTile += NPTILE) {

        // Fill aoI: weighted AO with derivatives (aow) for iTile atoms
        for (int k = lid; k < PT_ATOM_SIZE; k += WGS_VMAT) {
            int il = k & (NATILE - 1);
            int ip = k >> LOG_NATILE;
            int ia = (iTile << LOG_NATILE) + il;
            int g = gTile + ip;
            int base = il << LOG_MAX_AO_ATOM;
            for (int a = 0; a < MAX_AO_ATOM; a++) aoI[ip][base + a] = 0.0f;
            if (g < ngrids && ia < natoms) {
                float4 d = coords[g] - atom_coords[ia];
                float w0 = wv[g], wx = wv[ngrids + g], wy = wv[2*ngrids + g], wz = wv[3*ngrids + g];
                fill_atom_aow_gga(l_i_ns[il], &l_i_ir[il][0], d, w0, wx, wy, wz, base, aoI[ip], r0, du, nrad, rad_node, radial_l);
            }
        }

        // Fill aoJ: plain AO values (no derivatives) for jTile atoms
        for (int k = lid; k < PT_ATOM_SIZE; k += WGS_VMAT) {
            int jl = k & (NATILE - 1);
            int ip = k >> LOG_NATILE;
            int ja = (jTile << LOG_NATILE) + jl;
            int g = gTile + ip;
            int base = jl << LOG_MAX_AO_ATOM;
            for (int b = 0; b < MAX_AO_ATOM; b++) aoJ[ip][base + b] = 0.0f;
            if (g < ngrids && ja < natoms) {
                float4 d = coords[g] - atom_coords[ja];
                fill_atom_ao_lda(l_j_ns[jl], &l_j_ir[jl][0], d, base, aoJ[ip], r0, du, nrad, rad_node, radial_l);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int t = 0; t < QPT; t++) {
            int q = lid + t * WGS_VMAT;
            if (q >= VBLK_SIZE) continue;
            int iao_l, jao_l, il, jl, a, b;
            decode_q_vmat(q, &iao_l, &jao_l, &il, &jl, &a, &b);
            int ia = (iTile << LOG_NATILE) + il;
            int ja = (jTile << LOG_NATILE) + jl;
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
        int iao_l, jao_l, il, jl, a, b;
        decode_q_vmat(q, &iao_l, &jao_l, &il, &jl, &a, &b);
        int ia = (iTile << LOG_NATILE) + il;
        int ja = (jTile << LOG_NATILE) + jl;
        if (ia < natoms && ja < natoms && a < atom_nao[ia] && b < atom_nao[ja]) {
            int iao = atom_ao0[ia] + a;
            int jao = atom_ao0[ja] + b;
            vmat[iao * ncart + jao] = acc[t];
        }
    }
}

// ============================================================
// Single atom-pair kernels (NATILE=1 specialization, fixed AO layout)
// Host selects these when TileConfig.NATILE == 1.
// ============================================================

#define PAIR_AO_TILE MAX_AO_ATOM
#define PAIR_BLK_SIZE (PAIR_AO_TILE * PAIR_AO_TILE)
#define PAIR_QPT ((PAIR_BLK_SIZE + WGS_VMAT - 1) / WGS_VMAT)

inline void load_single_atom_meta(int ia, int natoms, __global const int *atom_radial_offset, __global const int *atom_radial_list, __local int *ns, __local int *ir_out)
{
    if (ia >= natoms) { *ns = 0; return; }
    int off = atom_radial_offset[ia];
    *ns = atom_radial_offset[ia + 1] - off;
    for (int s = 0; s < *ns; s++) ir_out[s] = atom_radial_list[off + s];
}

inline void load_single_atom_meta_l(int ia, int natoms, __global const int *atom_radial_offset, __global const int *atom_radial_list, __global const int *radial_l, __local int *ns, __local int *ir_out, __local int *l_out)
{
    if (ia >= natoms) { *ns = 0; return; }
    int off = atom_radial_offset[ia];
    *ns = atom_radial_offset[ia + 1] - off;
    for (int s = 0; s < *ns; s++) {
        int ir = atom_radial_list[off + s];
        ir_out[s] = ir;
        l_out[s] = radial_l[ir];
    }
}

// ---- Hermite AO grid projection (setup / precomp path) ----
// One WG per grid-point tile (NPTILE threads).  Outer loop over atoms; per atom
// collaborative __local radial cache wfR[NPTILE][MAX_SHELL], then each thread
// writes Cartesian AOs via eval_radial_cart*.  No DM — simpler than OTF rho/vmat.

__kernel void eval_ao_hermite_cart_tiled(
    __global const float4 *coords,
    __global const float4 *atom_coords,
    __global const float2 *rad_node,
    __global const int *radial_l,
    __global const int *radial_cart0,
    __global const int *atom_radial_offset,
    __global const int *atom_radial_list,
    __global float *ao,
    float r0, float du,
    int nrad, int ncart, int ngrids, int natoms)
{
    int ip = get_local_id(0);
    int lid = ip;
    int gTile = get_group_id(0);
    int g = gTile * NPTILE + ip;

    __local int l_ns;
    __local int l_ir[MAX_SHELL];
    __local int l_l[MAX_SHELL];
    __local float wfR[NPTILE][MAX_SHELL];

    for (int ja = 0; ja < natoms; ja++) {
        if (lid == 0) load_single_atom_meta_l(ja, natoms, atom_radial_offset, atom_radial_list, radial_l, &l_ns, l_ir, l_l);
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = lid; k < NPTILE * MAX_SHELL; k += NPTILE) {
            int pp = k / MAX_SHELL;
            int gj = gTile * NPTILE + pp;
            for (int s = 0; s < MAX_SHELL; s++) wfR[pp][s] = 0.0f;
            if (gj < ngrids && ja < natoms) {
                float4 dj = coords[gj] - atom_coords[ja];
                float t, t1m; int ik;
                hermite_map_point(dj, r0, du, nrad, &t, &t1m, &ik);
                for (int s = 0; s < l_ns; s++)
                    wfR[pp][s] = hermite_eval_ir(t, t1m, du, ik, l_ir[s], nrad, rad_node);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (g < ngrids && ja < natoms) {
            float4 d = coords[g] - atom_coords[ja];
            for (int s = 0; s < l_ns; s++)
                eval_radial_cart(d, l_l[s], g * ncart + radial_cart0[l_ir[s]], wfR[ip][s], ncart, ao);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void eval_ao_hermite_cart_deriv1_tiled(
    __global const float4 *coords,
    __global const float4 *atom_coords,
    __global const float2 *rad_node,
    __global const int *radial_l,
    __global const int *radial_cart0,
    __global const int *atom_radial_offset,
    __global const int *atom_radial_list,
    __global float *ao0,
    __global float *ao1,
    __global float *ao2,
    __global float *ao3,
    float r0, float du,
    int nrad, int ncart, int ngrids, int natoms)
{
    int ip = get_local_id(0);
    int lid = ip;
    int gTile = get_group_id(0);
    int g = gTile * NPTILE + ip;

    __local int l_ns;
    __local int l_ir[MAX_SHELL];
    __local int l_l[MAX_SHELL];
    __local float wfR[NPTILE][MAX_SHELL];
    __local float dwfR[NPTILE][MAX_SHELL];

    for (int ja = 0; ja < natoms; ja++) {
        if (lid == 0) load_single_atom_meta_l(ja, natoms, atom_radial_offset, atom_radial_list, radial_l, &l_ns, l_ir, l_l);
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = lid; k < NPTILE * MAX_SHELL; k += NPTILE) {
            int pp = k / MAX_SHELL;
            int gj = gTile * NPTILE + pp;
            for (int s = 0; s < MAX_SHELL; s++) { wfR[pp][s] = 0.0f; dwfR[pp][s] = 0.0f; }
            if (gj < ngrids && ja < natoms) {
                float4 dj = coords[gj] - atom_coords[ja];
                float rj = sqrt(dj.x*dj.x + dj.y*dj.y + dj.z*dj.z);
                float t, t1m; int ik;
                hermite_map_point(dj, r0, du, nrad, &t, &t1m, &ik);
                for (int s = 0; s < l_ns; s++) {
                    wfR[pp][s] = hermite_eval_ir(t, t1m, du, ik, l_ir[s], nrad, rad_node);
                    dwfR[pp][s] = hermite_eval_deriv_ir(t, t1m, du, ik, l_ir[s], nrad, rad_node) / (rj + r0);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (g < ngrids && ja < natoms) {
            float4 d = coords[g] - atom_coords[ja];
            float r = sqrt(d.x*d.x + d.y*d.y + d.z*d.z);
            float invr = r > 1.0e-20f ? 1.0f / r : 0.0f;
            for (int s = 0; s < l_ns; s++)
                eval_radial_cart_deriv1(d, l_l[s], g * ncart + radial_cart0[l_ir[s]], wfR[ip][s], dwfR[ip][s], invr, ncart, ao0, ao1, ao2, ao3);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

// [ngrids,nao] row-major ao -> [nao,ngrids] chi (coalesced ρ/vmat layout)
__kernel void transpose_ao_to_chi(
    __global const float *ao,
    __global float *chi,
    int nao, int ngrids)
{
    int g = get_global_id(0);
    int i = get_global_id(1);
    if (g >= ngrids || i >= nao) return;
    chi[i * ngrids + g] = ao[g * nao + i];
}

// Precompute R(ir,g) and dR/dr(ir,g) for radial_precomp ρ/vmat. Layout: out[ir*ngrids+g].
// Global: (ceil(ngrids/NPTILE), nradial), local: (NPTILE, 1) — coalesced writes along g.
__kernel void build_radial_on_grid_tiled(
    __global const float4 *coords,
    __global const float4 *atom_coords,
    __global const int *radial_atom,
    __global const float2 *rad_node,
    __global float *rad_val,
    __global float *rad_dr,
    float r0, float du,
    int nrad, int nradial, int ngrids)
{
    int ip = get_local_id(0);
    int gTile = get_group_id(0);
    int ir = get_group_id(1);
    int g = gTile * NPTILE + ip;
    if (ir >= nradial || g >= ngrids) return;

    int ia = radial_atom[ir];
    float4 d = coords[g] - atom_coords[ia];
    float r = sqrt(d.x*d.x + d.y*d.y + d.z*d.z);
    float t, t1m; int ik;
    hermite_map_point(d, r0, du, nrad, &t, &t1m, &ik);
    float R = hermite_eval_ir(t, t1m, du, ik, ir, nrad, rad_node);
    float dR_du = hermite_eval_deriv_ir(t, t1m, du, ik, ir, nrad, rad_node);
    int idx = ir * ngrids + g;
    rad_val[idx] = R;
    rad_dr[idx] = dR_du / (r + r0);
}

inline float contract_rho_pair_lda(float *Ri, float *Rj, __local float dm_blk[MAX_AO_ATOM][MAX_AO_ATOM], int ns_i, const int *l_i, int ns_j, const int *l_j, float4 di, float4 dj)
{
    float acc = 0.0f;
    int iao_off = 0;
    for (int si = 0; si < ns_i; si++) {
        float fi[6]; int ni = unfold_shell(l_i[si], Ri[si], di, fi);
        int jao_off = 0;
        for (int sj = 0; sj < ns_j; sj++) {
            float fj[6]; int nj = unfold_shell(l_j[sj], Rj[sj], dj, fj);
            for (int ai = 0; ai < ni; ai++) {
                float tmp = 0.0f;
                for (int aj = 0; aj < nj; aj++) tmp += dm_blk[iao_off + ai][jao_off + aj] * fj[aj];
                acc += fi[ai] * tmp;
            }
            jao_off += nj;
        }
        iao_off += ni;
    }
    return acc;
}

inline void contract_rho_pair_gga(float *Ri, float *dRi, float *Rj, float *dRj, __local float dm_blk[MAX_AO_ATOM][MAX_AO_ATOM], int ns_i, const int *l_i, int ns_j, const int *l_j, float4 di, float4 dj, float invr_i, float invr_j, float *rho_val, float *gx_val, float *gy_val, float *gz_val)
{
    *rho_val = 0.0f; *gx_val = 0.0f; *gy_val = 0.0f; *gz_val = 0.0f;
    int iao_off = 0;
    for (int si = 0; si < ns_i; si++) {
        float4 fi[6];
        int ni = unfold_shell_deriv_f4(l_i[si], Ri[si], dRi[si], di, invr_i, fi);
        int jao_off = 0;
        for (int sj = 0; sj < ns_j; sj++) {
            float4 fj[6];
            int nj = unfold_shell_deriv_f4(l_j[sj], Rj[sj], dRj[sj], dj, invr_j, fj);
            for (int ai = 0; ai < ni; ai++) {
                float4 t = (float4)(0.0f);
                for (int aj = 0; aj < nj; aj++) t += dm_blk[iao_off + ai][jao_off + aj] * fj[aj];
                accum_rho_gga_f4(fi[ai], t, rho_val, gx_val, gy_val, gz_val);
            }
            jao_off += nj;
        }
        iao_off += ni;
    }
}

// One workgroup per grid-point tile; loops all atom pairs (ia, ja).
__kernel void rho_lda_pair(
    __global const float4 *coords,
    __global const float4 *atom_coords,
    __global const float2 *rad_node,
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
    int lid = ip;
    int gTile = get_group_id(0);
    int g = gTile * NPTILE + ip;

    __local float wfRj[NPTILE][MAX_SHELL];
    __local float dm_blk[MAX_AO_ATOM][MAX_AO_ATOM];
    __local int l_j_ns;
    __local int l_j_ir[MAX_SHELL];
    __local int l_j_l[MAX_SHELL];

    float rho_priv = 0.0f;
    int i_ir[MAX_SHELL];
    int i_l[MAX_SHELL];
    float Ri[MAX_SHELL];

    for (int ja = 0; ja < natoms; ja++) {
        if (lid == 0) load_single_atom_meta_l(ja, natoms, atom_radial_offset, atom_radial_list, radial_l, &l_j_ns, l_j_ir, l_j_l);
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = lid; k < NPTILE * MAX_SHELL; k += NPTILE) {
            int pp = k / MAX_SHELL;
            int gj = gTile * NPTILE + pp;
            for (int s = 0; s < MAX_SHELL; s++) wfRj[pp][s] = 0.0f;
            if (gj < ngrids && ja < natoms) {
                float t, t1m; int ik;
                hermite_map_point(coords[gj] - atom_coords[ja], r0, du, nrad, &t, &t1m, &ik);
                for (int s = 0; s < l_j_ns; s++)
                    wfRj[pp][s] = hermite_eval_ir(t, t1m, du, ik, l_j_ir[s], nrad, rad_node);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int ia = 0; ia < natoms; ia++) {
            int ns_i = 0;
            float4 di = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
            float t_i = 0.0f, t1m_i = 0.0f; int ik_i = 0;
            for (int s = 0; s < MAX_SHELL; s++) Ri[s] = 0.0f;
            if (g < ngrids && ia < natoms) {
                ns_i = load_atom_ir_l(ia, natoms, atom_radial_offset, atom_radial_list, radial_l, i_ir, i_l);
                di = coords[g] - atom_coords[ia];
                hermite_map_point(di, r0, du, nrad, &t_i, &t1m_i, &ik_i);
                eval_radials_slice(ns_i, i_ir, t_i, t1m_i, du, ik_i, nrad, rad_node, Ri);
            }

            for (int k = lid; k < PAIR_BLK_SIZE; k += NPTILE) {
                int a = k >> LOG_MAX_AO_ATOM;
                int b = k & (MAX_AO_ATOM - 1);
                float v = 0.0f;
                if (ia < natoms && ja < natoms && a < atom_nao[ia] && b < atom_nao[ja])
                    v = dm_cart[(atom_ao0[ia] + a) * ncart + (atom_ao0[ja] + b)];
                dm_blk[a][b] = v;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            if (g < ngrids && ia < natoms) {
                float4 dj = coords[g] - atom_coords[ja];
                rho_priv += contract_rho_pair_lda(Ri, wfRj[ip], dm_blk, ns_i, i_l, l_j_ns, l_j_l, di, dj);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (g < ngrids) rho[g] = rho_priv;
}

__kernel void rho_gga_pair(
    __global const float4 *coords,
    __global const float4 *atom_coords,
    __global const float2 *rad_node,
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
    int lid = ip;
    int gTile = get_group_id(0);
    int g = gTile * NPTILE + ip;

    __local float wfRj[NPTILE][MAX_SHELL];
    __local float dwfRj[NPTILE][MAX_SHELL];
    __local float dm_blk[MAX_AO_ATOM][MAX_AO_ATOM];
    __local int l_j_ns;
    __local int l_j_ir[MAX_SHELL];
    __local int l_j_l[MAX_SHELL];

    float rho_priv = 0.0f, gx_priv = 0.0f, gy_priv = 0.0f, gz_priv = 0.0f;
    int i_ir[MAX_SHELL];
    int i_l[MAX_SHELL];
    float Ri[MAX_SHELL];
    float dRi[MAX_SHELL];

    for (int ja = 0; ja < natoms; ja++) {
        if (lid == 0) load_single_atom_meta_l(ja, natoms, atom_radial_offset, atom_radial_list, radial_l, &l_j_ns, l_j_ir, l_j_l);
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = lid; k < NPTILE * MAX_SHELL; k += NPTILE) {
            int pp = k / MAX_SHELL;
            int gj = gTile * NPTILE + pp;
            for (int s = 0; s < MAX_SHELL; s++) { wfRj[pp][s] = 0.0f; dwfRj[pp][s] = 0.0f; }
            if (gj < ngrids && ja < natoms) {
                float4 dj = coords[gj] - atom_coords[ja];
                float rj = sqrt(dj.x*dj.x + dj.y*dj.y + dj.z*dj.z);
                float t, t1m; int ik;
                hermite_map_point(dj, r0, du, nrad, &t, &t1m, &ik);
                for (int s = 0; s < l_j_ns; s++) {
                    wfRj[pp][s] = hermite_eval_ir(t, t1m, du, ik, l_j_ir[s], nrad, rad_node);
                    dwfRj[pp][s] = hermite_eval_deriv_ir(t, t1m, du, ik, l_j_ir[s], nrad, rad_node) / (rj + r0);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int ia = 0; ia < natoms; ia++) {
            int ns_i = 0;
            float4 di = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
            float invr_i = 0.0f;
            float t_i = 0.0f, t1m_i = 0.0f; int ik_i = 0;
            float r_i = 0.0f;
            for (int s = 0; s < MAX_SHELL; s++) { Ri[s] = 0.0f; dRi[s] = 0.0f; }
            if (g < ngrids && ia < natoms) {
                ns_i = load_atom_ir_l(ia, natoms, atom_radial_offset, atom_radial_list, radial_l, i_ir, i_l);
                di = coords[g] - atom_coords[ia];
                r_i = sqrt(di.x*di.x + di.y*di.y + di.z*di.z);
                invr_i = r_i > 1e-20f ? 1.0f/r_i : 0.0f;
                hermite_map_point(di, r0, du, nrad, &t_i, &t1m_i, &ik_i);
                eval_radials_slice_deriv(ns_i, i_ir, t_i, t1m_i, du, ik_i, nrad, r_i, r0, rad_node, Ri, dRi);
            }

            for (int k = lid; k < PAIR_BLK_SIZE; k += NPTILE) {
                int a = k >> LOG_MAX_AO_ATOM;
                int b = k & (MAX_AO_ATOM - 1);
                float v = 0.0f;
                if (ia < natoms && ja < natoms && a < atom_nao[ia] && b < atom_nao[ja])
                    v = dm_cart[(atom_ao0[ia] + a) * ncart + (atom_ao0[ja] + b)];
                dm_blk[a][b] = v;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            if (g < ngrids && ia < natoms) {
                float4 dj = coords[g] - atom_coords[ja];
                float rj = sqrt(dj.x*dj.x + dj.y*dj.y + dj.z*dj.z);
                float invr_j = rj > 1e-20f ? 1.0f/rj : 0.0f;
                float rv, gv1, gv2, gv3;
                contract_rho_pair_gga(Ri, dRi, wfRj[ip], dwfRj[ip], dm_blk, ns_i, i_l, l_j_ns, l_j_l, di, dj, invr_i, invr_j, &rv, &gv1, &gv2, &gv3);
                rho_priv += rv; gx_priv += gv1; gy_priv += gv2; gz_priv += gv3;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (g < ngrids) {
        rho[g] = rho_priv;
        rho[ngrids + g] = gx_priv;
        rho[2*ngrids + g] = gy_priv;
        rho[3*ngrids + g] = gz_priv;
    }
}

// One workgroup per atom pair (ia, ja); LDA writes symmetric off-diagonal blocks.
__kernel void vmat_lda_pair(
    __global const float4 *coords,
    __global const float4 *atom_coords,
    __global const float2 *rad_node,
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
    int ia = get_group_id(0);
    int ja = get_group_id(1);
    if (ja < ia) return;

    __local float aoI[NPTILE][MAX_AO_ATOM];
    __local float aoJ[NPTILE][MAX_AO_ATOM];
    __local int l_i_ns;
    __local int l_j_ns;
    __local int l_i_ir[MAX_SHELL];
    __local int l_j_ir[MAX_SHELL];

    if (lid == 0) {
        load_single_atom_meta(ia, natoms, atom_radial_offset, atom_radial_list, &l_i_ns, l_i_ir);
        load_single_atom_meta(ja, natoms, atom_radial_offset, atom_radial_list, &l_j_ns, l_j_ir);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float acc[PAIR_QPT];
    for (int t = 0; t < PAIR_QPT; t++) acc[t] = 0.0f;

    for (int gTile = 0; gTile < ngrids; gTile += NPTILE) {
        for (int k = lid; k < NPTILE; k += WGS_VMAT) {
            int ip = k;
            int g = gTile + ip;
            for (int a = 0; a < MAX_AO_ATOM; a++) aoI[ip][a] = 0.0f;
            if (g < ngrids && ia < natoms)
                fill_atom_ao_lda(l_i_ns, l_i_ir, coords[g] - atom_coords[ia], 0, aoI[ip], r0, du, nrad, rad_node, radial_l);
        }
        for (int k = lid; k < NPTILE; k += WGS_VMAT) {
            int ip = k;
            int g = gTile + ip;
            for (int b = 0; b < MAX_AO_ATOM; b++) aoJ[ip][b] = 0.0f;
            if (g < ngrids && ja < natoms)
                fill_atom_ao_lda(l_j_ns, l_j_ir, coords[g] - atom_coords[ja], 0, aoJ[ip], r0, du, nrad, rad_node, radial_l);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int t = 0; t < PAIR_QPT; t++) {
            int q = lid + t * WGS_VMAT;
            if (q >= PAIR_BLK_SIZE) continue;
            int a = q >> LOG_MAX_AO_ATOM;
            int b = q & (MAX_AO_ATOM - 1);
            if (ia >= natoms || ja >= natoms || a >= atom_nao[ia] || b >= atom_nao[ja]) continue;
            float s = 0.0f;
            for (int ip = 0; ip < NPTILE; ip++) {
                int g = gTile + ip;
                if (g >= ngrids) continue;
                s += wv[g] * aoI[ip][a] * aoJ[ip][b];
            }
            acc[t] += s;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int t = 0; t < PAIR_QPT; t++) {
        int q = lid + t * WGS_VMAT;
        if (q >= PAIR_BLK_SIZE) continue;
        int a = q >> LOG_MAX_AO_ATOM;
        int b = q & (MAX_AO_ATOM - 1);
        if (ia < natoms && ja < natoms && a < atom_nao[ia] && b < atom_nao[ja]) {
            int iao = atom_ao0[ia] + a;
            int jao = atom_ao0[ja] + b;
            vmat[iao * ncart + jao] = acc[t];
            if (ia != ja) vmat[jao * ncart + iao] = acc[t];
        }
    }
}

__kernel void vmat_gga_pair(
    __global const float4 *coords,
    __global const float4 *atom_coords,
    __global const float2 *rad_node,
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
    int ia = get_group_id(0);
    int ja = get_group_id(1);

    __local float aoI[NPTILE][MAX_AO_ATOM];
    __local float aoJ[NPTILE][MAX_AO_ATOM];
    __local int l_i_ns;
    __local int l_j_ns;
    __local int l_i_ir[MAX_SHELL];
    __local int l_j_ir[MAX_SHELL];

    if (lid == 0) {
        load_single_atom_meta(ia, natoms, atom_radial_offset, atom_radial_list, &l_i_ns, l_i_ir);
        load_single_atom_meta(ja, natoms, atom_radial_offset, atom_radial_list, &l_j_ns, l_j_ir);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float acc[PAIR_QPT];
    for (int t = 0; t < PAIR_QPT; t++) acc[t] = 0.0f;

    for (int gTile = 0; gTile < ngrids; gTile += NPTILE) {
        for (int k = lid; k < NPTILE; k += WGS_VMAT) {
            int ip = k;
            int g = gTile + ip;
            for (int a = 0; a < MAX_AO_ATOM; a++) aoI[ip][a] = 0.0f;
            if (g < ngrids && ia < natoms) {
                float w0 = wv[g], wx = wv[ngrids + g], wy = wv[2*ngrids + g], wz = wv[3*ngrids + g];
                fill_atom_aow_gga(l_i_ns, l_i_ir, coords[g] - atom_coords[ia], w0, wx, wy, wz, 0, aoI[ip], r0, du, nrad, rad_node, radial_l);
            }
        }
        for (int k = lid; k < NPTILE; k += WGS_VMAT) {
            int ip = k;
            int g = gTile + ip;
            for (int b = 0; b < MAX_AO_ATOM; b++) aoJ[ip][b] = 0.0f;
            if (g < ngrids && ja < natoms)
                fill_atom_ao_lda(l_j_ns, l_j_ir, coords[g] - atom_coords[ja], 0, aoJ[ip], r0, du, nrad, rad_node, radial_l);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int t = 0; t < PAIR_QPT; t++) {
            int q = lid + t * WGS_VMAT;
            if (q >= PAIR_BLK_SIZE) continue;
            int a = q >> LOG_MAX_AO_ATOM;
            int b = q & (MAX_AO_ATOM - 1);
            if (ia >= natoms || ja >= natoms || a >= atom_nao[ia] || b >= atom_nao[ja]) continue;
            float s = 0.0f;
            for (int ip = 0; ip < NPTILE; ip++) {
                int g = gTile + ip;
                if (g >= ngrids) continue;
                s += aoI[ip][a] * aoJ[ip][b];
            }
            acc[t] += s;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int t = 0; t < PAIR_QPT; t++) {
        int q = lid + t * WGS_VMAT;
        if (q >= PAIR_BLK_SIZE) continue;
        int a = q >> LOG_MAX_AO_ATOM;
        int b = q & (MAX_AO_ATOM - 1);
        if (ia < natoms && ja < natoms && a < atom_nao[ia] && b < atom_nao[ja]) {
            int iao = atom_ao0[ia] + a;
            int jao = atom_ao0[ja] + b;
            vmat[iao * ncart + jao] = acc[t];
        }
    }
}

// ============================================================
// Precomputed GTO — OTF-mirror pair gather (1 launch rho + 1 launch vmat)
// rho: one WG per grid tile, NPTILE threads, atom-pair loop with aoJ[NPTILE][MAX_AO_ATOM]
//      in __local (same pattern as rho_lda_pair / rho_gga_pair Hermite kernels).
// vmat: pair gather (vmat_*_precomp_pair), same as Hermite vmat_gga_pair.
// fused GEMM kernels below kept for fused=gemm fallback only.
// ============================================================
#define KTILE 32

__kernel void rho_lda_precomp_pair(
    __global const float *ao0,
    __global const float *dm,
    __global const int *atom_ao0,
    __global const int *atom_nao,
    __global float *rho,
    int nao, int ngrids, int natoms)
{
    int ip = get_local_id(0);
    int gTile = get_group_id(0);
    int g = gTile * NPTILE + ip;

    __local float aoJ[NPTILE][MAX_AO_ATOM];
    __local float dm_blk[MAX_AO_ATOM][MAX_AO_ATOM];

    float rho_priv = 0.0f;

    for (int ja = 0; ja < natoms; ja++) {
        for (int k = ip; k < NPTILE * MAX_AO_ATOM; k += NPTILE) {
            int pp = k / MAX_AO_ATOM;
            int b = k & (MAX_AO_ATOM - 1);
            int gj = gTile * NPTILE + pp;
            aoJ[pp][b] = 0.0f;
            if (gj < ngrids && ja < natoms && b < atom_nao[ja]) {
                int gbase = gj * nao;
                aoJ[pp][b] = ao0[gbase + atom_ao0[ja] + b];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int ia = 0; ia < natoms; ia++) {
            for (int kk = ip; kk < PAIR_BLK_SIZE; kk += NPTILE) {
                int a = kk >> LOG_MAX_AO_ATOM;
                int b = kk & (MAX_AO_ATOM - 1);
                float v = 0.0f;
                if (ia < natoms && ja < natoms && a < atom_nao[ia] && b < atom_nao[ja])
                    v = dm[(atom_ao0[ia] + a) * nao + (atom_ao0[ja] + b)];
                dm_blk[a][b] = v;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            if (g < ngrids && ia < natoms && ja < natoms) {
                int gbase = g * nao;
                int i0 = atom_ao0[ia];
                for (int a = 0; a < atom_nao[ia]; a++) {
                    float ai = ao0[gbase + i0 + a];
                    for (int b = 0; b < atom_nao[ja]; b++)
                        rho_priv += ai * dm_blk[a][b] * aoJ[ip][b];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (g < ngrids) rho[g] = rho_priv;
}

__kernel void rho_gga_precomp_pair(
    __global const float *ao0,
    __global const float *ao1,
    __global const float *ao2,
    __global const float *ao3,
    __global const float *dm,
    __global const int *atom_ao0,
    __global const int *atom_nao,
    __global float *rho,
    int nao, int ngrids, int natoms)
{
    int ip = get_local_id(0);
    int gTile = get_group_id(0);
    int g = gTile * NPTILE + ip;

    __local float aoJ0[NPTILE][MAX_AO_ATOM];
    __local float aoJ1[NPTILE][MAX_AO_ATOM];
    __local float aoJ2[NPTILE][MAX_AO_ATOM];
    __local float aoJ3[NPTILE][MAX_AO_ATOM];
    __local float dm_blk[MAX_AO_ATOM][MAX_AO_ATOM];

    float rho_priv = 0.0f, gx_priv = 0.0f, gy_priv = 0.0f, gz_priv = 0.0f;

    for (int ja = 0; ja < natoms; ja++) {
        for (int k = ip; k < NPTILE * MAX_AO_ATOM; k += NPTILE) {
            int pp = k / MAX_AO_ATOM;
            int b = k & (MAX_AO_ATOM - 1);
            int gj = gTile * NPTILE + pp;
            aoJ0[pp][b] = 0.0f; aoJ1[pp][b] = 0.0f;
            aoJ2[pp][b] = 0.0f; aoJ3[pp][b] = 0.0f;
            if (gj < ngrids && ja < natoms && b < atom_nao[ja]) {
                int idx = gj * nao + atom_ao0[ja] + b;
                aoJ0[pp][b] = ao0[idx];
                aoJ1[pp][b] = ao1[idx];
                aoJ2[pp][b] = ao2[idx];
                aoJ3[pp][b] = ao3[idx];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int ia = 0; ia < natoms; ia++) {
            for (int kk = ip; kk < PAIR_BLK_SIZE; kk += NPTILE) {
                int a = kk >> LOG_MAX_AO_ATOM;
                int b = kk & (MAX_AO_ATOM - 1);
                float v = 0.0f;
                if (ia < natoms && ja < natoms && a < atom_nao[ia] && b < atom_nao[ja])
                    v = dm[(atom_ao0[ia] + a) * nao + (atom_ao0[ja] + b)];
                dm_blk[a][b] = v;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            if (g < ngrids && ia < natoms && ja < natoms) {
                int gbase = g * nao;
                int i0 = atom_ao0[ia];
                for (int a = 0; a < atom_nao[ia]; a++) {
                    int idx = gbase + i0 + a;
                    float v0 = ao0[idx];
                    float v1 = ao1[idx];
                    float v2 = ao2[idx];
                    float v3 = ao3[idx];
                    for (int b = 0; b < atom_nao[ja]; b++) {
                        float d = dm_blk[a][b];
                        float j0v = aoJ0[ip][b];
                        float j1v = aoJ1[ip][b];
                        float j2v = aoJ2[ip][b];
                        float j3v = aoJ3[ip][b];
                        rho_priv += v0 * d * j0v;
                        gx_priv += v0 * d * j1v + v1 * d * j0v;
                        gy_priv += v0 * d * j2v + v2 * d * j0v;
                        gz_priv += v0 * d * j3v + v3 * d * j0v;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (g < ngrids) {
        rho[g] = rho_priv;
        rho[ngrids + g] = gx_priv;
        rho[2 * ngrids + g] = gy_priv;
        rho[3 * ngrids + g] = gz_priv;
    }
}

// ============================================================
// Experimental optimized rho projection kernels (rho only)
// ============================================================
//
// These kernels are intentionally not wired into the Python harness here.
// They define the GPU-side contract for two future precomputed-rho paths.
//
// Common execution model:
//   - One workgroup owns one contiguous grid tile of NPTILE grid points.
//   - One thread owns one grid point g = gTile*NPTILE + ip.
//   - The thread accumulates all atom-pair contributions to rho[g].
//   - DM atom-pair blocks are cooperatively gathered into __local dm_blk.
//   - J-atom AO/radial values for the whole grid tile are gathered into
//     __local memory to amortize global reads over all i-atoms.
//
// Why this layout:
//   Current precomputed AO buffers are [iG, iAO] row-major. For a workgroup
//   loading one AO component for consecutive grid points, addresses differ by
//   stride nao, so the warp/wavefront performs a gather. The optimized full-AO
//   kernel below requires pre-transposed AO buffers [iAO, iG], i.e.
//
//       chi_c[ iAO * ngrids + iG ]
//
//   With this layout, all threads in the workgroup load the same iAO at
//   consecutive iG, producing coalesced global memory transactions.
//
// Workgroup size:
//   local = (NPTILE,), typically NPTILE=64.  This keeps the ownership model
//   simple: thread ip owns rho[g].  Local memory is modest:
//     full-AO GGA: 4*NPTILE*MAX_AO_ATOM floats + MAX_AO_ATOM^2 floats
//     radial GGA : 2*NPTILE*MAX_SHELL  floats + MAX_AO_ATOM^2 floats
//
// Future screening hook:
//   The outer loops over ja/ia currently run over all atoms.  A later harness
//   can replace these loops with per-gTile active atom lists from grid_screen.py
//   without changing the local memory strategy.

__kernel void rho_gga_precomp_coalesced_pair(
    __global const float *chi0,     // [nao, ngrids], chi0[mu*ngrids + g]
    __global const float *chi1,     // [nao, ngrids], d chi / dx
    __global const float *chi2,     // [nao, ngrids], d chi / dy
    __global const float *chi3,     // [nao, ngrids], d chi / dz
    __global const float *dm,       // [nao, nao], row-major
    __global const int *atom_ao0,   // first AO of each atom in chi/dm
    __global const int *atom_nao,   // number of AOs on each atom, <= MAX_AO_ATOM
    __global float *rho,            // [4, ngrids] = rho, gx, gy, gz
    int nao, int ngrids, int natoms)
{
    int ip = get_local_id(0);
    int gTile = get_group_id(0);
    int g = gTile * NPTILE + ip;

    __local float aoJ0[NPTILE][MAX_AO_ATOM];
    __local float aoJ1[NPTILE][MAX_AO_ATOM];
    __local float aoJ2[NPTILE][MAX_AO_ATOM];
    __local float aoJ3[NPTILE][MAX_AO_ATOM];
    __local float dm_blk[MAX_AO_ATOM][MAX_AO_ATOM];

    float rho_priv = 0.0f;
    float gx_priv = 0.0f;
    float gy_priv = 0.0f;
    float gz_priv = 0.0f;

    for (int ja = 0; ja < natoms; ja++) {
        int j0 = atom_ao0[ja];
        int nja = atom_nao[ja];

        // Gather J-atom AO values for this whole grid tile.
        // For fixed b, threads ip=0..NPTILE-1 read chi[(j0+b)*ngrids + g],
        // which is contiguous in global memory.
        for (int b = 0; b < MAX_AO_ATOM; b++) {
            float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
            if (g < ngrids && b < nja) {
                int idx = (j0 + b) * ngrids + g;
                v0 = chi0[idx];
                v1 = chi1[idx];
                v2 = chi2[idx];
                v3 = chi3[idx];
            }
            aoJ0[ip][b] = v0;
            aoJ1[ip][b] = v1;
            aoJ2[ip][b] = v2;
            aoJ3[ip][b] = v3;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int ia = 0; ia < natoms; ia++) {
            int i0 = atom_ao0[ia];
            int nia = atom_nao[ia];

            // Gather one atom-pair DM tile into local memory.  Threads stride
            // across the 16x16 tile; global DM reads are mostly contiguous in b.
            for (int kk = ip; kk < PAIR_BLK_SIZE; kk += NPTILE) {
                int a = kk >> LOG_MAX_AO_ATOM;
                int b = kk & (MAX_AO_ATOM - 1);
                float d = 0.0f;
                if (a < nia && b < nja)
                    d = dm[(i0 + a) * nao + (j0 + b)];
                dm_blk[a][b] = d;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            if (g < ngrids) {
                // I-atom values are used only by this thread/grid point, so
                // keep them in registers.  For fixed a, all workgroup threads
                // again read consecutive chi[(i0+a)*ngrids + g] addresses.
                for (int a = 0; a < nia; a++) {
                    int idxI = (i0 + a) * ngrids + g;
                    float i0v = chi0[idxI];
                    float i1v = chi1[idxI];
                    float i2v = chi2[idxI];
                    float i3v = chi3[idxI];

                    for (int b = 0; b < nja; b++) {
                        float d = dm_blk[a][b];
                        float j0v = aoJ0[ip][b];
                        float j1v = aoJ1[ip][b];
                        float j2v = aoJ2[ip][b];
                        float j3v = aoJ3[ip][b];
                        rho_priv += i0v * d * j0v;
                        gx_priv  += i0v * d * j1v + i1v * d * j0v;
                        gy_priv  += i0v * d * j2v + i2v * d * j0v;
                        gz_priv  += i0v * d * j3v + i3v * d * j0v;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (g < ngrids) {
        rho[g] = rho_priv;
        rho[ngrids + g] = gx_priv;
        rho[2 * ngrids + g] = gy_priv;
        rho[3 * ngrids + g] = gz_priv;
    }
}

inline void contract_rho_pair_gga_radial_precomp(
    float *Ri,
    float *dRi,
    __local float *Rj,
    __local float *dRj,
    __local float dm_blk[MAX_AO_ATOM][MAX_AO_ATOM],
    int ns_i,
    const int *l_i,
    int ns_j,
    __local const int *l_j,
    float4 di,
    float4 dj,
    float invr_i,
    float invr_j,
    float *rho_val,
    float *gx_val,
    float *gy_val,
    float *gz_val)
{
    *rho_val = 0.0f;
    *gx_val = 0.0f;
    *gy_val = 0.0f;
    *gz_val = 0.0f;

    int iao_off = 0;
    for (int si = 0; si < ns_i; si++) {
        float4 fi[6];
        int ni = unfold_shell_deriv_f4(l_i[si], Ri[si], dRi[si], di, invr_i, fi);
        int jao_off = 0;
        for (int sj = 0; sj < ns_j; sj++) {
            float4 fj[6];
            int nj = unfold_shell_deriv_f4(l_j[sj], Rj[sj], dRj[sj], dj, invr_j, fj);
            for (int ai = 0; ai < ni; ai++) {
                float4 t = (float4)(0.0f);
                for (int aj = 0; aj < nj; aj++) {
                    float d = dm_blk[iao_off + ai][jao_off + aj];
                    t += d * fj[aj];
                }
                accum_rho_gga_f4(fi[ai], t, rho_val, gx_val, gy_val, gz_val);
            }
            jao_off += nj;
        }
        iao_off += ni;
    }
}

__kernel void rho_gga_radial_precomp_pair(
    __global const float4 *coords,       // [ngrids], xyz in Bohr
    __global const float4 *atom_coords,  // [natoms], xyz in Bohr
    __global const float *rad_val,       // [nradial, ngrids], R(ir,g)
    __global const float *rad_dr,        // [nradial, ngrids], dR/dr(ir,g)
    __global const int *radial_l,        // [nradial]
    __global const int *atom_radial_offset,
    __global const int *atom_radial_list,
    __global const float *dm_cart,       // [ncart, ncart], Cartesian AO basis
    __global const int *atom_ao0,
    __global const int *atom_nao,
    __global float *rho,                 // [4, ngrids] = rho, gx, gy, gz
    int ncart, int ngrids, int natoms)
{
    int ip = get_local_id(0);
    int lid = ip;
    int gTile = get_group_id(0);
    int g = gTile * NPTILE + ip;

    __local float wfRj[NPTILE][MAX_SHELL];
    __local float dwfRj[NPTILE][MAX_SHELL];
    __local float dm_blk[MAX_AO_ATOM][MAX_AO_ATOM];
    __local int l_j_ns;
    __local int l_j_ir[MAX_SHELL];
    __local int l_j_l[MAX_SHELL];

    float rho_priv = 0.0f;
    float gx_priv = 0.0f;
    float gy_priv = 0.0f;
    float gz_priv = 0.0f;

    int i_ir[MAX_SHELL];
    int i_l[MAX_SHELL];
    float Ri[MAX_SHELL];
    float dRi[MAX_SHELL];

    for (int ja = 0; ja < natoms; ja++) {
        if (lid == 0)
            load_single_atom_meta_l(ja, natoms, atom_radial_offset, atom_radial_list, radial_l, &l_j_ns, l_j_ir, l_j_l);
        barrier(CLK_LOCAL_MEM_FENCE);

        // Gather precomputed radial values for the J atom.  The layout is
        // rad_val[ir*ngrids + g], so for fixed shell ir the workgroup reads
        // consecutive grid points.  This keeps the bandwidth-heavy radial
        // precomputed path coalesced while avoiding storage of full Cartesian
        // derivatives.
        for (int s = 0; s < MAX_SHELL; s++) {
            float rv = 0.0f;
            float dr = 0.0f;
            if (g < ngrids && s < l_j_ns) {
                int idx = l_j_ir[s] * ngrids + g;
                rv = rad_val[idx];
                dr = rad_dr[idx];
            }
            wfRj[ip][s] = rv;
            dwfRj[ip][s] = dr;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int ia = 0; ia < natoms; ia++) {
            int ns_i = 0;
            float4 di = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
            float invr_i = 0.0f;
            for (int s = 0; s < MAX_SHELL; s++) {
                Ri[s] = 0.0f;
                dRi[s] = 0.0f;
            }

            if (g < ngrids && ia < natoms) {
                ns_i = load_atom_ir_l(ia, natoms, atom_radial_offset, atom_radial_list, radial_l, i_ir, i_l);
                di = coords[g] - atom_coords[ia];
                float ri = sqrt(di.x*di.x + di.y*di.y + di.z*di.z);
                invr_i = ri > 1e-20f ? 1.0f / ri : 0.0f;
                for (int s = 0; s < ns_i; s++) {
                    int idx = i_ir[s] * ngrids + g;
                    Ri[s] = rad_val[idx];
                    dRi[s] = rad_dr[idx];
                }
            }

            for (int kk = ip; kk < PAIR_BLK_SIZE; kk += NPTILE) {
                int a = kk >> LOG_MAX_AO_ATOM;
                int b = kk & (MAX_AO_ATOM - 1);
                float d = 0.0f;
                if (ia < natoms && ja < natoms && a < atom_nao[ia] && b < atom_nao[ja])
                    d = dm_cart[(atom_ao0[ia] + a) * ncart + (atom_ao0[ja] + b)];
                dm_blk[a][b] = d;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            if (g < ngrids && ia < natoms) {
                float4 dj = coords[g] - atom_coords[ja];
                float rj = sqrt(dj.x*dj.x + dj.y*dj.y + dj.z*dj.z);
                float invr_j = rj > 1e-20f ? 1.0f / rj : 0.0f;
                float rv, gv1, gv2, gv3;
                contract_rho_pair_gga_radial_precomp(Ri, dRi, wfRj[ip], dwfRj[ip], dm_blk,
                                                     ns_i, i_l, l_j_ns, l_j_l, di, dj,
                                                     invr_i, invr_j, &rv, &gv1, &gv2, &gv3);
                rho_priv += rv;
                gx_priv += gv1;
                gy_priv += gv2;
                gz_priv += gv3;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (g < ngrids) {
        rho[g] = rho_priv;
        rho[ngrids + g] = gx_priv;
        rho[2 * ngrids + g] = gy_priv;
        rho[3 * ngrids + g] = gz_priv;
    }
}

// ============================================================
// Experimental optimized vmat assembly kernels
// ============================================================
//
// These mirror vmat_gga_precomp_pair / vmat_gga_pair but consume the optimized
// rho-side layouts:
//   - coalesced full AO: chi_c[mu*ngrids + g]
//   - radial precomp   : rad_val[ir*ngrids + g], rad_dr[ir*ngrids + g]
//
// GGA convention matches existing kernels: compute the one-sided matrix
//     V_ij = sum_g (wv0*phi_i + wvx*dphi_i/dx + ...)(g) * phi_j(g)
// and let the host add V + V.T.

__kernel void vmat_gga_precomp_coalesced_pair(
    __global const float *chi0,     // [nao, ngrids], chi0[mu*ngrids + g]
    __global const float *chi1,     // [nao, ngrids], d chi / dx
    __global const float *chi2,     // [nao, ngrids], d chi / dy
    __global const float *chi3,     // [nao, ngrids], d chi / dz
    __global const float *wv,       // [4, ngrids]
    __global const int *atom_ao0,
    __global const int *atom_nao,
    __global float *vmat,           // [nao, nao], one-sided GGA
    int nao, int ngrids, int natoms)
{
    int lid = get_local_id(1);
    int ia = get_group_id(0);
    int ja = get_group_id(1);

    __local float aowI[NPTILE][MAX_AO_ATOM];
    __local float aoJ[NPTILE][MAX_AO_ATOM];

    float acc[PAIR_QPT];
    for (int t = 0; t < PAIR_QPT; t++) acc[t] = 0.0f;

    int i0 = (ia < natoms) ? atom_ao0[ia] : 0;
    int j0 = (ja < natoms) ? atom_ao0[ja] : 0;
    int nia = (ia < natoms) ? atom_nao[ia] : 0;
    int nja = (ja < natoms) ? atom_nao[ja] : 0;

    for (int gTile = 0; gTile < ngrids; gTile += NPTILE) {
        for (int k = lid; k < NPTILE; k += WGS_VMAT) {
            int ip = k;
            int g = gTile + ip;
            for (int a = 0; a < MAX_AO_ATOM; a++) {
                aowI[ip][a] = 0.0f;
                aoJ[ip][a] = 0.0f;
            }
            if (g < ngrids && ia < natoms && ja < natoms) {
                float w0 = wv[g];
                float wx = wv[ngrids + g];
                float wy = wv[2 * ngrids + g];
                float wz = wv[3 * ngrids + g];
                for (int a = 0; a < nia; a++) {
                    int idx = (i0 + a) * ngrids + g;
                    aowI[ip][a] = w0 * chi0[idx] + wx * chi1[idx] + wy * chi2[idx] + wz * chi3[idx];
                }
                for (int b = 0; b < nja; b++)
                    aoJ[ip][b] = chi0[(j0 + b) * ngrids + g];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int t = 0; t < PAIR_QPT; t++) {
            int q = lid + t * WGS_VMAT;
            if (q >= PAIR_BLK_SIZE) continue;
            int a = q >> LOG_MAX_AO_ATOM;
            int b = q & (MAX_AO_ATOM - 1);
            if (a >= nia || b >= nja) continue;
            float s = 0.0f;
            for (int ip = 0; ip < NPTILE; ip++) {
                int g = gTile + ip;
                if (g >= ngrids) continue;
                s += aowI[ip][a] * aoJ[ip][b];
            }
            acc[t] += s;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int t = 0; t < PAIR_QPT; t++) {
        int q = lid + t * WGS_VMAT;
        if (q >= PAIR_BLK_SIZE) continue;
        int a = q >> LOG_MAX_AO_ATOM;
        int b = q & (MAX_AO_ATOM - 1);
        if (ia < natoms && ja < natoms && a < nia && b < nja)
            vmat[(i0 + a) * nao + (j0 + b)] = acc[t];
    }
}

inline void fill_atom_ao_radial_precomp(
    int ns,
    __local const int *ir_list,
    __local const int *l_list,
    __global const float *rad_val,
    int g,
    int ngrids,
    float4 d,
    __local float *ao)
{
    int ao0 = 0;
    for (int s = 0; s < ns; s++) {
        float f[6];
        int ir = ir_list[s];
        int n = unfold_shell(l_list[s], rad_val[ir * ngrids + g], d, f);
        for (int a = 0; a < n; a++) ao[ao0 + a] = f[a];
        ao0 += n;
    }
}

inline void fill_atom_aow_gga_radial_precomp(
    int ns,
    __local const int *ir_list,
    __local const int *l_list,
    __global const float *rad_val,
    __global const float *rad_dr,
    int g,
    int ngrids,
    float4 d,
    float w0,
    float wx,
    float wy,
    float wz,
    __local float *aow)
{
    float r = sqrt(d.x*d.x + d.y*d.y + d.z*d.z);
    float invr = r > 1e-20f ? 1.0f / r : 0.0f;
    int ao0 = 0;
    for (int s = 0; s < ns; s++) {
        float f0[6], f1[6], f2[6], f3[6];
        int ir = ir_list[s];
        int n = unfold_shell_deriv(l_list[s], rad_val[ir * ngrids + g], rad_dr[ir * ngrids + g], d, invr, f0, f1, f2, f3);
        for (int a = 0; a < n; a++)
            aow[ao0 + a] = w0*f0[a] + wx*f1[a] + wy*f2[a] + wz*f3[a];
        ao0 += n;
    }
}

__kernel void vmat_gga_radial_precomp_pair(
    __global const float4 *coords,       // [ngrids], xyz in Bohr
    __global const float4 *atom_coords,  // [natoms], xyz in Bohr
    __global const float *rad_val,       // [nradial, ngrids], R(ir,g)
    __global const float *rad_dr,        // [nradial, ngrids], dR/dr(ir,g)
    __global const int *radial_l,        // [nradial]
    __global const int *atom_radial_offset,
    __global const int *atom_radial_list,
    __global const int *atom_ao0,        // Cartesian atom AO offsets
    __global const int *atom_nao,        // Cartesian AOs per atom
    __global const float *wv,            // [4, ngrids]
    __global float *vmat,                // [ncart, ncart], one-sided GGA
    int ncart, int ngrids, int natoms)
{
    int lid = get_local_id(1);
    int ia = get_group_id(0);
    int ja = get_group_id(1);

    __local float aowI[NPTILE][MAX_AO_ATOM];
    __local float aoJ[NPTILE][MAX_AO_ATOM];
    __local int l_i_ns;
    __local int l_j_ns;
    __local int l_i_ir[MAX_SHELL];
    __local int l_j_ir[MAX_SHELL];
    __local int l_i_l[MAX_SHELL];
    __local int l_j_l[MAX_SHELL];

    if (lid == 0) {
        load_single_atom_meta_l(ia, natoms, atom_radial_offset, atom_radial_list, radial_l, &l_i_ns, l_i_ir, l_i_l);
        load_single_atom_meta_l(ja, natoms, atom_radial_offset, atom_radial_list, radial_l, &l_j_ns, l_j_ir, l_j_l);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float acc[PAIR_QPT];
    for (int t = 0; t < PAIR_QPT; t++) acc[t] = 0.0f;

    int i0 = (ia < natoms) ? atom_ao0[ia] : 0;
    int j0 = (ja < natoms) ? atom_ao0[ja] : 0;
    int nia = (ia < natoms) ? atom_nao[ia] : 0;
    int nja = (ja < natoms) ? atom_nao[ja] : 0;

    for (int gTile = 0; gTile < ngrids; gTile += NPTILE) {
        for (int k = lid; k < NPTILE; k += WGS_VMAT) {
            int ip = k;
            int g = gTile + ip;
            for (int a = 0; a < MAX_AO_ATOM; a++) {
                aowI[ip][a] = 0.0f;
                aoJ[ip][a] = 0.0f;
            }
            if (g < ngrids && ia < natoms && ja < natoms) {
                float w0 = wv[g];
                float wx = wv[ngrids + g];
                float wy = wv[2 * ngrids + g];
                float wz = wv[3 * ngrids + g];
                fill_atom_aow_gga_radial_precomp(l_i_ns, l_i_ir, l_i_l, rad_val, rad_dr,
                                                 g, ngrids, coords[g] - atom_coords[ia],
                                                 w0, wx, wy, wz, aowI[ip]);
                fill_atom_ao_radial_precomp(l_j_ns, l_j_ir, l_j_l, rad_val,
                                            g, ngrids, coords[g] - atom_coords[ja], aoJ[ip]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int t = 0; t < PAIR_QPT; t++) {
            int q = lid + t * WGS_VMAT;
            if (q >= PAIR_BLK_SIZE) continue;
            int a = q >> LOG_MAX_AO_ATOM;
            int b = q & (MAX_AO_ATOM - 1);
            if (a >= nia || b >= nja) continue;
            float s = 0.0f;
            for (int ip = 0; ip < NPTILE; ip++) {
                int g = gTile + ip;
                if (g >= ngrids) continue;
                s += aowI[ip][a] * aoJ[ip][b];
            }
            acc[t] += s;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int t = 0; t < PAIR_QPT; t++) {
        int q = lid + t * WGS_VMAT;
        if (q >= PAIR_BLK_SIZE) continue;
        int a = q >> LOG_MAX_AO_ATOM;
        int b = q & (MAX_AO_ATOM - 1);
        if (ia < natoms && ja < natoms && a < nia && b < nja)
            vmat[(i0 + a) * ncart + (j0 + b)] = acc[t];
    }
}

__kernel void rho_lda_precomp_tiled(
    __global const float *ao0,
    __global const float *dm,
    __global const int *atom_ao0,
    __global const int *atom_nao,
    __global float *rho,
    int nao, int ngrids, int natoms)
{
    int ip = get_local_id(0);
    int il = get_local_id(1);
    int lid = il * NPTILE + ip;
    int gTile = get_group_id(0);
    int g = gTile * NPTILE + ip;

    __local float aoJ[NPTILE][NATILE][MAX_AO_ATOM];
    __local float dm_blk[NATILE][NATILE][MAX_AO_ATOM][MAX_AO_ATOM];
    __local float psum[WGS_TILED];

    float rho_priv = 0.0f;
    int n_iTiles = (natoms + NATILE - 1) / NATILE;
    float aoI_tile[MAX_ITILE][MAX_AO_ATOM];
    int n_i_ao[MAX_ITILE];

    for (int it = 0; it < n_iTiles; it++) {
        int ia = (it << LOG_NATILE) + il;
        n_i_ao[it] = 0;
        for (int a = 0; a < MAX_AO_ATOM; a++) aoI_tile[it][a] = 0.0f;
        if (g < ngrids && ia < natoms) {
            n_i_ao[it] = atom_nao[ia];
            int gbase = g * nao;
            int i0 = atom_ao0[ia];
            for (int a = 0; a < n_i_ao[it]; a++)
                aoI_tile[it][a] = ao0[gbase + i0 + a];
        }
    }

    for (int jTile = 0; jTile < natoms; jTile += NATILE) {
        for (int k = lid; k < PT_ATOM_SIZE; k += WGS_TILED) {
            int jj = k & (NATILE - 1);
            int pp = k >> LOG_NATILE;
            int gj = gTile * NPTILE + pp;
            int ja = jTile + jj;
            for (int a = 0; a < MAX_AO_ATOM; a++) aoJ[pp][jj][a] = 0.0f;
            if (gj < ngrids && ja < natoms) {
                int gbase = gj * nao;
                int j0 = atom_ao0[ja];
                for (int b = 0; b < atom_nao[ja]; b++)
                    aoJ[pp][jj][b] = ao0[gbase + j0 + b];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int it = 0; it < n_iTiles; it++) {
            int ia = (it << LOG_NATILE) + il;
            for (int k = lid; k < DM_TILE_SIZE; k += WGS_TILED) {
                int ab = k & (AO_BLK - 1);
                int pq = k >> LOG_AO_BLK;
                int a = ab >> LOG_MAX_AO_ATOM;
                int b = ab & (MAX_AO_ATOM - 1);
                int ii2 = pq >> LOG_NATILE;
                int jj2 = pq & (NATILE - 1);
                int ia2 = (it << LOG_NATILE) + ii2, ja2 = jTile + jj2;
                float v = 0.0f;
                if (ia2 < natoms && ja2 < natoms && a < atom_nao[ia2] && b < atom_nao[ja2]) {
                    v = dm[(atom_ao0[ia2] + a) * nao + (atom_ao0[ja2] + b)];
                }
                dm_blk[ii2][jj2][a][b] = v;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            if (g < ngrids && ia < natoms) {
                for (int jl = 0; jl < NATILE; jl++) {
                    int ja = jTile + jl;
                    if (ja >= natoms) continue;
                    int nja = atom_nao[ja];
                    for (int a = 0; a < n_i_ao[it]; a++) {
                        float ai = aoI_tile[it][a];
                        for (int b = 0; b < nja; b++)
                            rho_priv += ai * dm_blk[il][jl][a][b] * aoJ[ip][jl][b];
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    psum[lid] = rho_priv;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (il == 0 && g < ngrids) {
        float s = 0.0f;
        for (int k = 0; k < NATILE; k++) s += psum[k * NPTILE + ip];
        rho[g] = s;
    }
}

__kernel void rho_gga_precomp_tiled(
    __global const float *ao0,
    __global const float *ao1,
    __global const float *ao2,
    __global const float *ao3,
    __global const float *dm,
    __global const int *atom_ao0,
    __global const int *atom_nao,
    __global float *rho,
    int nao, int ngrids, int natoms)
{
    int ip = get_local_id(0);
    int il = get_local_id(1);
    int lid = il * NPTILE + ip;
    int gTile = get_group_id(0);
    int g = gTile * NPTILE + ip;

    __local float aoJ0[NPTILE][NATILE][MAX_AO_ATOM];
    __local float aoJ1[NPTILE][NATILE][MAX_AO_ATOM];
    __local float aoJ2[NPTILE][NATILE][MAX_AO_ATOM];
    __local float aoJ3[NPTILE][NATILE][MAX_AO_ATOM];
    __local float dm_blk[NATILE][NATILE][MAX_AO_ATOM][MAX_AO_ATOM];
    __local float psum_rho[WGS_TILED];
    __local float psum_gx[WGS_TILED];
    __local float psum_gy[WGS_TILED];
    __local float psum_gz[WGS_TILED];

    float rho_priv = 0.0f, gx_priv = 0.0f, gy_priv = 0.0f, gz_priv = 0.0f;
    int n_iTiles = (natoms + NATILE - 1) / NATILE;
    float aoI0[MAX_ITILE][MAX_AO_ATOM];
    float aoI1[MAX_ITILE][MAX_AO_ATOM];
    float aoI2[MAX_ITILE][MAX_AO_ATOM];
    float aoI3[MAX_ITILE][MAX_AO_ATOM];
    int n_i_ao[MAX_ITILE];

    for (int it = 0; it < n_iTiles; it++) {
        int ia = (it << LOG_NATILE) + il;
        n_i_ao[it] = 0;
        for (int a = 0; a < MAX_AO_ATOM; a++) {
            aoI0[it][a] = 0.0f; aoI1[it][a] = 0.0f;
            aoI2[it][a] = 0.0f; aoI3[it][a] = 0.0f;
        }
        if (g < ngrids && ia < natoms) {
            n_i_ao[it] = atom_nao[ia];
            int gbase = g * nao;
            int i0 = atom_ao0[ia];
            for (int a = 0; a < n_i_ao[it]; a++) {
                int idx = gbase + i0 + a;
                aoI0[it][a] = ao0[idx];
                aoI1[it][a] = ao1[idx];
                aoI2[it][a] = ao2[idx];
                aoI3[it][a] = ao3[idx];
            }
        }
    }

    for (int jTile = 0; jTile < natoms; jTile += NATILE) {
        for (int k = lid; k < PT_ATOM_SIZE; k += WGS_TILED) {
            int jj = k & (NATILE - 1);
            int pp = k >> LOG_NATILE;
            int gj = gTile * NPTILE + pp;
            int ja = jTile + jj;
            for (int a = 0; a < MAX_AO_ATOM; a++) {
                aoJ0[pp][jj][a] = 0.0f; aoJ1[pp][jj][a] = 0.0f;
                aoJ2[pp][jj][a] = 0.0f; aoJ3[pp][jj][a] = 0.0f;
            }
            if (gj < ngrids && ja < natoms) {
                int gbase = gj * nao;
                int j0 = atom_ao0[ja];
                for (int b = 0; b < atom_nao[ja]; b++) {
                    int idx = gbase + j0 + b;
                    aoJ0[pp][jj][b] = ao0[idx];
                    aoJ1[pp][jj][b] = ao1[idx];
                    aoJ2[pp][jj][b] = ao2[idx];
                    aoJ3[pp][jj][b] = ao3[idx];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int it = 0; it < n_iTiles; it++) {
            int ia = (it << LOG_NATILE) + il;
            for (int k = lid; k < DM_TILE_SIZE; k += WGS_TILED) {
                int ab = k & (AO_BLK - 1);
                int pq = k >> LOG_AO_BLK;
                int a = ab >> LOG_MAX_AO_ATOM;
                int b = ab & (MAX_AO_ATOM - 1);
                int ii2 = pq >> LOG_NATILE;
                int jj2 = pq & (NATILE - 1);
                int ia2 = (it << LOG_NATILE) + ii2, ja2 = jTile + jj2;
                float v = 0.0f;
                if (ia2 < natoms && ja2 < natoms && a < atom_nao[ia2] && b < atom_nao[ja2]) {
                    v = dm[(atom_ao0[ia2] + a) * nao + (atom_ao0[ja2] + b)];
                }
                dm_blk[ii2][jj2][a][b] = v;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            if (g < ngrids && ia < natoms) {
                for (int jl = 0; jl < NATILE; jl++) {
                    int ja = jTile + jl;
                    if (ja >= natoms) continue;
                    int nja = atom_nao[ja];
                    for (int a = 0; a < n_i_ao[it]; a++) {
                        float v0 = aoI0[it][a];
                        float v1 = aoI1[it][a];
                        float v2 = aoI2[it][a];
                        float v3 = aoI3[it][a];
                        for (int b = 0; b < nja; b++) {
                            float d = dm_blk[il][jl][a][b];
                            float j0v = aoJ0[ip][jl][b];
                            float j1v = aoJ1[ip][jl][b];
                            float j2v = aoJ2[ip][jl][b];
                            float j3v = aoJ3[ip][jl][b];
                            rho_priv += v0 * d * j0v;
                            gx_priv += v0 * d * j1v + v1 * d * j0v;
                            gy_priv += v0 * d * j2v + v2 * d * j0v;
                            gz_priv += v0 * d * j3v + v3 * d * j0v;
                        }
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    psum_rho[lid] = rho_priv; psum_gx[lid] = gx_priv;
    psum_gy[lid] = gy_priv;   psum_gz[lid] = gz_priv;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (il == 0 && g < ngrids) {
        float sr = 0.0f, sx = 0.0f, sy = 0.0f, sz = 0.0f;
        for (int k = 0; k < NATILE; k++) {
            sr += psum_rho[k * NPTILE + ip]; sx += psum_gx[k * NPTILE + ip];
            sy += psum_gy[k * NPTILE + ip]; sz += psum_gz[k * NPTILE + ip];
        }
        rho[g] = sr;
        rho[ngrids + g] = sx;
        rho[2 * ngrids + g] = sy;
        rho[3 * ngrids + g] = sz;
    }
}

__kernel void rho_lda_precomp_fused(
    __global const float *ao0,
    __global const float *dm,
    __global float *rho,
    int nao, int ngrids)
{
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int gTile = get_group_id(0);
    int nTiles = (nao + TILE - 1) / TILE;
    int nRowPanels = NPTILE / TILE;

    __local float ao_jt[TILE][TILE];
    __local float dm_tile[TILE][TILE];

    for (int rowPanel = 0; rowPanel < nRowPanels; rowPanel++) {
        int ip = rowPanel * TILE + tx;
        int g = gTile * NPTILE + ip;
        float ao_buf[128];
        float aodm_buf[128];
        if (g < ngrids) {
            for (int mu = 0; mu < nao; mu++)
                ao_buf[mu] = ao0[g * nao + mu];
        }
        for (int mu = 0; mu < nao; mu++)
            aodm_buf[mu] = 0.0f;

        for (int kt = 0; kt < nTiles; kt++) {
            for (int jt = 0; jt < nTiles; jt++) {
                if (g < ngrids && ty < TILE) {
                    int mu = jt * TILE + ty;
                    ao_jt[tx][ty] = (mu < nao) ? ao_buf[mu] : 0.0f;
                } else {
                    ao_jt[tx][ty] = 0.0f;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                int ii = kt * TILE + ty;
                int jj = jt * TILE + tx;
                dm_tile[ty][tx] = (ii < nao && jj < nao) ? dm[ii * nao + jj] : 0.0f;
                barrier(CLK_LOCAL_MEM_FENCE);
                if (g < ngrids && ii < nao) {
                    float s = 0.0f;
                    for (int t = 0; t < TILE; t++)
                        s += ao_jt[tx][t] * dm_tile[ty][t];
                    aodm_buf[ii] += s;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        if (g < ngrids && ty == 0) {
            float rho_v = 0.0f;
            for (int mu = 0; mu < nao; mu++)
                rho_v += ao_buf[mu] * aodm_buf[mu];
            rho[g] = rho_v;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void rho_gga_precomp_fused(
    __global const float *ao0,
    __global const float *ao1,
    __global const float *ao2,
    __global const float *ao3,
    __global const float *dm,
    __global float *rho,
    int nao, int ngrids)
{
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int gTile = get_group_id(0);
    int nTiles = (nao + TILE - 1) / TILE;
    int nRowPanels = NPTILE / TILE;

    __local float ao0_jt[TILE][TILE];
    __local float ao1_jt[TILE][TILE];
    __local float ao2_jt[TILE][TILE];
    __local float ao3_jt[TILE][TILE];
    __local float dm_tile[TILE][TILE];
    __local float aodm0_loc[TILE][TILE];
    __local float aodm1_loc[TILE][TILE];
    __local float aodm2_loc[TILE][TILE];
    __local float aodm3_loc[TILE][TILE];

    for (int rowPanel = 0; rowPanel < nRowPanels; rowPanel++) {
        int ip = rowPanel * TILE + tx;
        int g = gTile * NPTILE + ip;
        float ao0_buf[128], ao1_buf[128], ao2_buf[128], ao3_buf[128];
        if (g < ngrids) {
            for (int mu = 0; mu < nao; mu++) {
                int idx = g * nao + mu;
                ao0_buf[mu] = ao0[idx];
                ao1_buf[mu] = ao1[idx];
                ao2_buf[mu] = ao2[idx];
                ao3_buf[mu] = ao3[idx];
            }
        }
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

        for (int kt = 0; kt < nTiles; kt++) {
            if (ty < TILE) {
                aodm0_loc[tx][ty] = 0.0f; aodm1_loc[tx][ty] = 0.0f;
                aodm2_loc[tx][ty] = 0.0f; aodm3_loc[tx][ty] = 0.0f;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            for (int jt = 0; jt < nTiles; jt++) {
                if (g < ngrids && ty < TILE) {
                    int mu = jt * TILE + ty;
                    if (mu < nao) {
                        int idx = g * nao + mu;
                        ao0_jt[tx][ty] = ao0[idx];
                        ao1_jt[tx][ty] = ao1[idx];
                        ao2_jt[tx][ty] = ao2[idx];
                        ao3_jt[tx][ty] = ao3[idx];
                    } else {
                        ao0_jt[tx][ty] = 0.0f; ao1_jt[tx][ty] = 0.0f;
                        ao2_jt[tx][ty] = 0.0f; ao3_jt[tx][ty] = 0.0f;
                    }
                } else {
                    ao0_jt[tx][ty] = 0.0f; ao1_jt[tx][ty] = 0.0f;
                    ao2_jt[tx][ty] = 0.0f; ao3_jt[tx][ty] = 0.0f;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                int ii = kt * TILE + ty;
                int jj = jt * TILE + tx;
                dm_tile[ty][tx] = (ii < nao && jj < nao) ? dm[ii * nao + jj] : 0.0f;
                barrier(CLK_LOCAL_MEM_FENCE);
                if (g < ngrids && ii < nao) {
                    float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
                    for (int t = 0; t < TILE; t++) {
                        float d = dm_tile[ty][t];
                        v0 += ao0_jt[tx][t] * d;
                        v1 += ao1_jt[tx][t] * d;
                        v2 += ao2_jt[tx][t] * d;
                        v3 += ao3_jt[tx][t] * d;
                    }
                    aodm0_loc[tx][ty] += v0; aodm1_loc[tx][ty] += v1;
                    aodm2_loc[tx][ty] += v2; aodm3_loc[tx][ty] += v3;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (g < ngrids && ty == 0) {
                for (int t = 0; t < TILE; t++) {
                    int mu = kt * TILE + t;
                    if (mu >= nao) continue;
                    float f0 = ao0_buf[mu];
                    float f1 = ao1_buf[mu];
                    float f2 = ao2_buf[mu];
                    float f3 = ao3_buf[mu];
                    s0 += f0 * aodm0_loc[tx][t];
                    s1 += f0 * aodm1_loc[tx][t] + f1 * aodm0_loc[tx][t];
                    s2 += f0 * aodm2_loc[tx][t] + f2 * aodm0_loc[tx][t];
                    s3 += f0 * aodm3_loc[tx][t] + f3 * aodm0_loc[tx][t];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (g < ngrids && ty == 0) {
            rho[g] = s0;
            rho[ngrids + g] = s1;
            rho[2 * ngrids + g] = s2;
            rho[3 * ngrids + g] = s3;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void vmat_lda_precomp_pair(
    __global const float *ao0,
    __global const float *wv,
    __global const int *atom_ao0,
    __global const int *atom_nao,
    __global float *vmat,
    int nao, int ngrids, int natoms)
{
    int lid = get_local_id(1);
    int ia = get_group_id(0);
    int ja = get_group_id(1);
    __local float aoI[NPTILE][MAX_AO_ATOM];
    __local float aoJ[NPTILE][MAX_AO_ATOM];
    float acc[PAIR_QPT];
    for (int t = 0; t < PAIR_QPT; t++) acc[t] = 0.0f;

    for (int gTile = 0; gTile < ngrids; gTile += NPTILE) {
        for (int k = lid; k < NPTILE; k += WGS_VMAT) {
            int ip = k;
            int g = gTile + ip;
            for (int a = 0; a < MAX_AO_ATOM; a++) { aoI[ip][a] = 0.0f; aoJ[ip][a] = 0.0f; }
            if (g < ngrids && ia < natoms && ja < natoms) {
                int gbase = g * nao;
                int i0 = atom_ao0[ia];
                int j0 = atom_ao0[ja];
                for (int a = 0; a < atom_nao[ia]; a++)
                    aoI[ip][a] = wv[g] * ao0[gbase + i0 + a];
                for (int b = 0; b < atom_nao[ja]; b++)
                    aoJ[ip][b] = ao0[gbase + j0 + b];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int t = 0; t < PAIR_QPT; t++) {
            int q = lid + t * WGS_VMAT;
            if (q >= PAIR_BLK_SIZE) continue;
            int a = q >> LOG_MAX_AO_ATOM;
            int b = q & (MAX_AO_ATOM - 1);
            if (ia >= natoms || ja >= natoms || a >= atom_nao[ia] || b >= atom_nao[ja]) continue;
            float s = 0.0f;
            for (int ip = 0; ip < NPTILE; ip++) {
                int g = gTile + ip;
                if (g >= ngrids) continue;
                s += aoI[ip][a] * aoJ[ip][b];
            }
            acc[t] += s;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int t = 0; t < PAIR_QPT; t++) {
        int q = lid + t * WGS_VMAT;
        if (q >= PAIR_BLK_SIZE) continue;
        int a = q >> LOG_MAX_AO_ATOM;
        int b = q & (MAX_AO_ATOM - 1);
        if (ia < natoms && ja < natoms && a < atom_nao[ia] && b < atom_nao[ja]) {
            int iao = atom_ao0[ia] + a;
            int jao = atom_ao0[ja] + b;
            vmat[iao * nao + jao] = acc[t];
        }
    }
}

__kernel void vmat_gga_precomp_pair(
    __global const float *ao0,
    __global const float *ao1,
    __global const float *ao2,
    __global const float *ao3,
    __global const float *wv,
    __global const int *atom_ao0,
    __global const int *atom_nao,
    __global float *vmat,
    int nao, int ngrids, int natoms)
{
    int lid = get_local_id(1);
    int ia = get_group_id(0);
    int ja = get_group_id(1);
    __local float aoI[NPTILE][MAX_AO_ATOM];
    __local float aoJ[NPTILE][MAX_AO_ATOM];
    float acc[PAIR_QPT];
    for (int t = 0; t < PAIR_QPT; t++) acc[t] = 0.0f;

    for (int gTile = 0; gTile < ngrids; gTile += NPTILE) {
        for (int k = lid; k < NPTILE; k += WGS_VMAT) {
            int ip = k;
            int g = gTile + ip;
            for (int a = 0; a < MAX_AO_ATOM; a++) { aoI[ip][a] = 0.0f; aoJ[ip][a] = 0.0f; }
            if (g < ngrids && ia < natoms && ja < natoms) {
                int gbase = g * nao;
                float w0 = wv[g];
                float wx = wv[ngrids + g];
                float wy = wv[2 * ngrids + g];
                float wz = wv[3 * ngrids + g];
                int i0 = atom_ao0[ia];
                int j0 = atom_ao0[ja];
                for (int a = 0; a < atom_nao[ia]; a++) {
                    int idx = gbase + i0 + a;
                    aoI[ip][a] = w0 * ao0[idx] + wx * ao1[idx] + wy * ao2[idx] + wz * ao3[idx];
                }
                for (int b = 0; b < atom_nao[ja]; b++)
                    aoJ[ip][b] = ao0[gbase + j0 + b];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int t = 0; t < PAIR_QPT; t++) {
            int q = lid + t * WGS_VMAT;
            if (q >= PAIR_BLK_SIZE) continue;
            int a = q >> LOG_MAX_AO_ATOM;
            int b = q & (MAX_AO_ATOM - 1);
            if (ia >= natoms || ja >= natoms || a >= atom_nao[ia] || b >= atom_nao[ja]) continue;
            float s = 0.0f;
            for (int ip = 0; ip < NPTILE; ip++) {
                int g = gTile + ip;
                if (g >= ngrids) continue;
                s += aoI[ip][a] * aoJ[ip][b];
            }
            acc[t] += s;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int t = 0; t < PAIR_QPT; t++) {
        int q = lid + t * WGS_VMAT;
        if (q >= PAIR_BLK_SIZE) continue;
        int a = q >> LOG_MAX_AO_ATOM;
        int b = q & (MAX_AO_ATOM - 1);
        if (ia < natoms && ja < natoms && a < atom_nao[ia] && b < atom_nao[ja]) {
            int iao = atom_ao0[ia] + a;
            int jao = atom_ao0[ja] + b;
            vmat[iao * nao + jao] = acc[t];
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
