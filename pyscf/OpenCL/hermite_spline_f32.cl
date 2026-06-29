// Standalone f32 Hermite radial spline eval (study kernel; not wired into production AO path).
// order: 0=cubic float2(y,d)  1=quintic float4(y,d,c,pad)
// space: 0=u-mode (x=β·log1p(r/r0))  1=r-mode

inline int seg_lower(float x, __global const float *grid, int n)
{
    int lo = 0, hi = n - 2;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (grid[mid] <= x) lo = mid; else hi = mid - 1;
    }
    return lo;
}

inline float cubic_eval_f32(float t, float h, float y0, float y1, float d0, float d1)
{
    float t1m = t - 1.0f;
    float dy = y1 - y0;
    return y0 + t*t*(3.0f-2.0f*t)*dy + t*t1m*t1m*h*d0 + t*t*t1m*h*d1;
}

inline float cubic_deriv_f32(float t, float h, float y0, float y1, float d0, float d1)
{
    float t1m = t - 1.0f;
    float dy = y1 - y0;
    return (6.0f*t*(1.0f-t)*dy + (3.0f*t-1.0f)*t1m*h*d0 + t*(3.0f*t-2.0f)*h*d1) / h;
}

inline float quintic_eval_f32(float t, float h, float y0, float y1, float d0, float d1, float c0, float c1)
{
    float t2 = t*t, t3 = t2*t, t4 = t3*t, t5 = t4*t;
    float H00 = 1.0f - 10.0f*t3 + 15.0f*t4 - 6.0f*t5;
    float H10 = t - 6.0f*t3 + 8.0f*t4 - 3.0f*t5;
    float H20 = 0.5f*(t2 - 3.0f*t3 + 3.0f*t4 - t5);
    float H01 = 10.0f*t3 - 15.0f*t4 + 6.0f*t5;
    float H11 = -4.0f*t3 + 7.0f*t4 - 3.0f*t5;
    float H21 = 0.5f*(t3 - 2.0f*t4 + t5);
    float h2 = h*h;
    return H00*y0 + h*H10*d0 + h2*H20*c0 + H01*y1 + h*H11*d1 + h2*H21*c1;
}

inline float quintic_deriv_f32(float t, float h, float y0, float y1, float d0, float d1, float c0, float c1)
{
    float t2 = t*t, t3 = t2*t, t4 = t3*t;
    float dH00 = -30.0f*t2 + 60.0f*t3 - 30.0f*t4;
    float dH10 = 1.0f - 18.0f*t2 + 32.0f*t3 - 15.0f*t4;
    float dH20 = 0.5f*(2.0f*t - 9.0f*t2 + 12.0f*t3 - 5.0f*t4);
    float dH01 = 30.0f*t2 - 60.0f*t3 + 30.0f*t4;
    float dH11 = -12.0f*t2 + 28.0f*t3 - 15.0f*t4;
    float dH21 = 0.5f*(3.0f*t2 - 8.0f*t3 + 5.0f*t4);
    float h2 = h*h;
    return (dH00*y0 + h*dH10*d0 + h2*dH20*c0 + dH01*y1 + h*dH11*d1 + h2*dH21*c1) / h;
}

inline void eval_channel(
    float rq, float r0, float map_b, int space, int order,
    __global const float *x_grid, int n_nodes,
    __global const float *y, __global const float *d, __global const float *c,
    float *out_R, float *out_dR)
{
    float xq = (space == 0) ? map_b * log1p(rq / r0) : rq;
    int ik = seg_lower(xq, x_grid, n_nodes);
    float h = x_grid[ik + 1] - x_grid[ik];
    float t = clamp((xq - x_grid[ik]) / h, 0.0f, 1.0f);
    float y0 = y[ik], y1 = y[ik + 1];
    float d0 = d[ik], d1 = d[ik + 1];
    float R, dx;
    if (order == 0) {
        R = cubic_eval_f32(t, h, y0, y1, d0, d1);
        dx = cubic_deriv_f32(t, h, y0, y1, d0, d1);
    } else {
        float c0 = c[ik], c1 = c[ik + 1];
        R = quintic_eval_f32(t, h, y0, y1, d0, d1, c0, c1);
        dx = quintic_deriv_f32(t, h, y0, y1, d0, d1, c0, c1);
    }
    *out_R = R;
    if (space == 0)
        *out_dR = dx * map_b / (rq + r0);
    else
        *out_dR = dx;
}

__kernel void eval_radial_spline_f32(
    __global const float *r_query,
    __global const float *x_grid,
    __global const float *y,
    __global const float *d,
    __global const float *c,
    __global float *out_R,
    __global float *out_dR,
    int n_query,
    int n_nodes,
    float r0,
    float map_b,
    int order,
    int space)
{
    int iq = get_global_id(0);
    if (iq >= n_query) return;
    float rq = r_query[iq];
    float R, dR;
    eval_channel(rq, r0, map_b, space, order, x_grid, n_nodes, y, d, c, &R, &dR);
    out_R[iq] = R;
    out_dR[iq] = dR;
}
