
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <xmmintrin.h>

#include <windows.h>
#include <tchar.h>
#include <MMSystem.h>
#pragma comment(lib,"Winmm.lib")

typedef unsigned int IUINT32;

typedef __declspec(align(16)) struct { float m[4][4]; } matrix_t;
typedef __declspec(align(16)) struct { float x, y, z, w; } vector_t;
typedef vector_t point_t;

//#define CALL_CNT
//#define NORMAL_INTERP

#ifdef CALL_CNT
long vl = 0, vadd = 0, vsub = 0, vsca = 0, mtra = 0, mssca = 0, mrot = 0, mproj = 0, mlook = 0,
vdot = 0, vcross = 0, vinterp = 0, vnor = 0, madd = 0, msub = 0, mmul = 0, msca = 0, mapp = 0, mid = 0, mzero = 0;
#endif

int CMID(int x, int min, int max) { return (x < min) ? min : ((x > max) ? max : x); }
float saturate(float x) { return (x < 0.f) ? 0.f : ((x > 1.f) ? 1.f : x); }

// 计算插值：t 为 [0, 1] 之间的数值
float interp(float x1, float x2, float t) { return x1 + (x2 - x1) * t; }

// | v |
float vector_length(const vector_t *v) {
	float sq = v->x * v->x + v->y * v->y + v->z * v->z;
#ifdef CALL_CNT 
	vl++;
#endif
	return (float)sqrt(sq);
}

// z = x + y
vector_t* vector_add(vector_t *z, const vector_t *x, const vector_t *y) {
	/**/
	__m128 a = _mm_load_ps((float*)x);
	__m128 b = _mm_load_ps((float*)y);
	_mm_store_ps((float*)z, _mm_add_ps(a, b));

	/**/
	z->x = x->x + y->x;
	z->y = x->y + y->y;
	z->z = x->z + y->z;
	/**/
#ifdef CALL_CNT 
	vadd++;
#endif
	z->w = 1.f;
	return z;
}

void __stdcall sub_ps(vector_t*, const vector_t*, const vector_t*);
// z = x - y
vector_t* vector_sub(vector_t *z, const vector_t *x, const vector_t *y) {
	/* Win32 only *
	sub_ps(z, x, y);

	/* Win32 only *
	__m128 a = _mm_load_ps((float*) x);
	__m128 b = _mm_load_ps((float*) y);
	__m128 r;
	__asm {
	movaps xmm0, xmmword ptr [a]
	movaps xmm1, xmmword ptr [b]
	subps xmm0, xmm1
	movaps xmmword ptr [r], xmm0
	}
	_mm_store_ps((float*) z, r);

	/**/
	__m128 a = _mm_load_ps((float*)x);
	__m128 b = _mm_load_ps((float*)y);
	_mm_store_ps((float*)z, _mm_sub_ps(a, b));

	/**
	z->x = x->x - y->x;
	z->y = x->y - y->y;
	z->z = x->z - y->z;
	/**/
#ifdef CALL_CNT 
	vsub++;
#endif
	z->w = 1.f;
	return z;
}

// z = x * t
vector_t* vector_scale(vector_t *z, const vector_t *x, float t) {
#ifdef CALL_CNT 
	vsca++;
#endif
	z->x = x->x * t;
	z->y = x->y * t;
	z->z = x->z * t;
	z->w = 1.f;
	return z;
}
// 矢量点乘
float vector_dotproduct(const vector_t *x, const vector_t *y) {
#ifdef CALL_CNT 
	vdot++;
#endif
	return x->x * y->x + x->y * y->y + x->z * y->z;
}

// 矢量叉乘
vector_t* vector_crossproduct(vector_t *z, const vector_t *x, const vector_t *y) {
	float m1, m2, m3;
#ifdef CALL_CNT 
	vcross++;
#endif
	m1 = x->y * y->z - x->z * y->y;
	m2 = x->z * y->x - x->x * y->z;
	m3 = x->x * y->y - x->y * y->x;
	z->x = m1;
	z->y = m2;
	z->z = m3;
	z->w = 1.f;
	return z;
}

// 矢量插值，t取值 [0, 1]
vector_t* vector_interp(vector_t *z, const vector_t *x1, const vector_t *x2, float t) {
#ifdef CALL_CNT 
	vinterp++;
#endif
	z->x = interp(x1->x, x2->x, t);
	z->y = interp(x1->y, x2->y, t);
	z->z = interp(x1->z, x2->z, t);
	z->w = 1.f;
	return z;
}

// 矢量归一化
vector_t* vector_normalize(vector_t *v) {
	float length = vector_length(v);
#ifdef CALL_CNT 
	vnor++;
#endif
	if (length > 1e-6) {
		float inv = 1.f / length;
		v->x *= inv;
		v->y *= inv;
		v->z *= inv;
	}
	return v;
}

// c = a + b
matrix_t* matrix_add(matrix_t *c, const matrix_t *a, const matrix_t *b) {
	int i, j;
#ifdef CALL_CNT 
	madd++;
#endif
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++)
			c->m[i][j] = a->m[i][j] + b->m[i][j];
	}
	return c;
}

// c = a - b
matrix_t* matrix_sub(matrix_t *c, const matrix_t *a, const matrix_t *b) {
	int i, j;
#ifdef CALL_CNT 
	msub++;
#endif
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++)
			c->m[i][j] = a->m[i][j] - b->m[i][j];
	}
	return c;
}

#define SHUFFLE_PARAM(x, y, z, w) \
	((x) | (y << 2) | (z << 4) | (w << 6))
#define _mm_repx_ps(v) \
	_mm_shuffle_ps((v), (v), SHUFFLE_PARAM(0, 0, 0, 0))
#define _mm_repy_ps(v) \
	_mm_shuffle_ps((v), (v), SHUFFLE_PARAM(1, 1, 1, 1))
#define _mm_repz_ps(v) \
	_mm_shuffle_ps((v), (v), SHUFFLE_PARAM(2, 2, 2, 2))
#define _mm_repw_ps(v) \
	_mm_shuffle_ps((v), (v), SHUFFLE_PARAM(3, 3, 3, 3))
#define _mm_madd_ps(a, b, c) \
	_mm_add_ps(_mm_mul_ps((a), (b)), (c))

// c = a * b
matrix_t* matrix_mul(matrix_t *c, const matrix_t *a, const matrix_t *b) {
	/**/
	__m128 a0 = _mm_load_ps(a->m[0]);
	__m128 a1 = _mm_load_ps(a->m[1]);
	__m128 a2 = _mm_load_ps(a->m[2]);
	__m128 a3 = _mm_load_ps(a->m[3]);

	__m128 b0 = _mm_load_ps(b->m[0]);
	__m128 b1 = _mm_load_ps(b->m[1]);
	__m128 b2 = _mm_load_ps(b->m[2]);
	__m128 b3 = _mm_load_ps(b->m[3]);

	__m128 r;
	r = _mm_mul_ps(_mm_repx_ps(a0), b0);
	r = _mm_madd_ps(_mm_repy_ps(a0), b1, r);
	r = _mm_madd_ps(_mm_repz_ps(a0), b2, r);
	r = _mm_madd_ps(_mm_repw_ps(a0), b3, r);
	_mm_store_ps((float*)c->m[0], r);

	r = _mm_mul_ps(_mm_repx_ps(a1), b0);
	r = _mm_madd_ps(_mm_repy_ps(a1), b1, r);
	r = _mm_madd_ps(_mm_repz_ps(a1), b2, r);
	r = _mm_madd_ps(_mm_repw_ps(a1), b3, r);
	_mm_store_ps((float*)c->m[1], r);

	r = _mm_mul_ps(_mm_repx_ps(a2), b0);
	r = _mm_madd_ps(_mm_repy_ps(a2), b1, r);
	r = _mm_madd_ps(_mm_repz_ps(a2), b2, r);
	r = _mm_madd_ps(_mm_repw_ps(a2), b3, r);
	_mm_store_ps((float*)c->m[2], r);

	r = _mm_mul_ps(_mm_repx_ps(a3), b0);
	r = _mm_madd_ps(_mm_repy_ps(a3), b1, r);
	r = _mm_madd_ps(_mm_repz_ps(a3), b2, r);
	r = _mm_madd_ps(_mm_repw_ps(a3), b3, r);
	_mm_store_ps((float*)c->m[3], r);

	/**
	matrix_t z;
	int i, j;
	for (i = 0; i < 4; i++) {
	for (j = 0; j < 4; j++) {
	z.m[j][i] = (a->m[j][0] * b->m[0][i]) +
	(a->m[j][1] * b->m[1][i]) +
	(a->m[j][2] * b->m[2][i]) +
	(a->m[j][3] * b->m[3][i]);
	}
	}
	c[0] = z;
	/**/
#ifdef CALL_CNT 
	mmul++;
#endif
	return c;
}

//float matrix_determinant(const matrix_t* p) {
//	int r, c, m;
//	int lop = 0;
//	float result = 0;
//	float mid = 1;
//
//	for (m = 0; m < 4; m++) {
//		mid = 1;            //顺序求和, 主对角线元素相乘之和  
//		for (r = 0, c = m; r < 4; r++, c++)
//			mid = mid * p->m[r][c%4];
//		result += mid;
//	}
//	for (m = 0; m < 4; m++) {
//		mid = 1;            //逆序相减, 减去次对角线元素乘积  
//		for (r = 0, c = 3 - m + 4; r < 4; r++, c--)
//			mid = mid * p->m[r][c%4];
//		result -= mid;
//	}
//	return result;
//}

//float matrix_cofactor(const matrix_t* p, int m, int n) {  
//	int len;  
//	int i, j;  
//	float mid_result = 0;  
//	int sign = 1;  
//	float *p_creat, *p_mid;  
//
//	len = 9;            //k阶矩阵的代数余之式为k-1阶矩阵  
//	p_creat = (float*)calloc(len, sizeof(float)); //分配内存单元  
//	p_mid = p_creat;  
//	for (i = 0; i < 4; i++)  
//	{  
//		for (j = 0; j < 4; j++)  
//		{  
//			if (i != m && j != n) //将除第i行和第j列外的所有元素存储到以p_mid为首地址的内存单元  
//			{  
//				*p_mid++ = p->m[i][j];
//			}  
//		}  
//	}  
//	sign = (m + n) % 2 == 0 ? 1 : -1;    //代数余之式前面的正、负号  
//	mid_result = (float)sign*matrix_determinant(p_creat, k - 1);  
//	free(p_creat);  
//	return mid_result;  
//}  

float MatDet(float *p, int n)
{
	int r, c, m;
	int lop = 0;
	float result = 0;
	float mid = 1;

	if (n != 1)
	{
		lop = (n == 2) ? 1 : n;            //控制求和循环次数,若为2阶，则循环1次，否则为n次  
		for (m = 0; m < lop; m++)
		{
			mid = 1;            //顺序求和, 主对角线元素相乘之和  
			for (r = 0, c = m; r < n; r++, c++)
			{
				mid = mid * (*(p + r*n + c%n));
			}
			result += mid;
		}
		for (m = 0; m < lop; m++)
		{
			mid = 1;            //逆序相减, 减去次对角线元素乘积  
			for (r = 0, c = n - 1 - m + n; r < n; r++, c--)
			{
				mid = mid * (*(p + r*n + c%n));
			}
			result -= mid;
		}
	}
	else
		result = *p;
	return result;
}

//----------------------------------------------------------------------------  
//功能: 求k*k矩阵中元素A(m, n)的代数余之式  
//入口参数: k*k矩阵的首地址，矩阵元素A的下标m,n,矩阵行数k  
//返回值: k*k矩阵中元素A(m, n)的代数余之式  
//----------------------------------------------------------------------------  
float Creat_M(float *p, int m, int n, int k)
{
	int len;
	int i, j;
	float mid_result = 0;
	int sign = 1;
	float *p_creat, *p_mid;

	len = (k - 1)*(k - 1);            //k阶矩阵的代数余之式为k-1阶矩阵  
	p_creat = (float*)calloc(len, sizeof(float)); //分配内存单元  
	p_mid = p_creat;
	for (i = 0; i < k; i++)
	{
		for (j = 0; j < k; j++)
		{
			if (i != m && j != n) //将除第i行和第j列外的所有元素存储到以p_mid为首地址的内存单元  
			{
				*p_mid++ = *(p + i*k + j);
			}
		}
	}
	sign = (m + n) % 2 == 0 ? 1 : -1;    //代数余之式前面的正、负号  
	mid_result = (float)sign*MatDet(p_creat, k - 1);
	free(p_creat);
	return mid_result;
}

//------------------------------------------------------------------  
//功能: 采用部分主元的高斯消去法求方阵A的逆矩阵B  
//入口参数: 输入方阵，输出方阵，方阵阶数  
//返回值: true or false  
//-------------------------------------------------------------------  
char gauss(float** A, float** B) {
	int i, j, k;
	float max, temp;
	float t[4][4];                //临时矩阵  
								  //将A矩阵存放在临时矩阵t[n][n]中  
	for (i = 0; i < 4; i++)
	{
		for (j = 0; j < 4; j++)
		{
			t[i][j] = A[i][j];
		}
	}
	//初始化B矩阵为单位阵  
	for (i = 0; i < 4; i++)
	{
		for (j = 0; j < 4; j++)
		{
			B[i][j] = (i == j) ? (float)1 : 0;
		}
	}
	for (i = 0; i < 4; i++)
	{
		//寻找主元  
		max = t[i][i];
		k = i;
		for (j = i + 1; j < 4; j++)
		{
			if (fabs(t[j][i]) > fabs(max))
			{
				max = t[j][i];
				k = j;
			}
		}
		//如果主元所在行不是第i行，进行行交换  
		if (k != i)
		{
			for (j = 0; j < 4; j++)
			{
				temp = t[i][j];
				t[i][j] = t[k][j];
				t[k][j] = temp;
				//B伴随交换  
				temp = B[i][j];
				B[i][j] = B[k][j];
				B[k][j] = temp;
			}
		}
		//判断主元是否为0, 若是, 则矩阵A不是满秩矩阵,不存在逆矩阵  
		if (t[i][i] == 0) return 0;
		//消去A的第i列除去i行以外的各行元素  
		temp = t[i][i];
		for (j = 0; j < 4; j++)
		{
			t[i][j] = t[i][j] / temp;        //主对角线上的元素变为1  
			B[i][j] = B[i][j] / temp;        //伴随计算  
		}
		for (j = 0; j < 4; j++)        //第0行->第n行  
		{
			if (j != i)                //不是第i行  
			{
				temp = t[j][i];
				for (k = 0; k < 4; k++)        //第j行元素 - i行元素*j列i行元素  
				{
					t[j][k] = t[j][k] - t[i][k] * temp;
					B[j][k] = B[j][k] - B[i][k] * temp;
				}
			}
		}
	}
	return 1;
}

// c = a * f
matrix_t* matrix_scale(matrix_t *c, const matrix_t *a, float f) {
	int i, j;
#ifdef CALL_CNT 
	msca++;
#endif
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++)
			c->m[i][j] = a->m[i][j] * f;
	}
	return c;
}

// y = x * m
vector_t* matrix_apply(vector_t *y, const vector_t *x, const matrix_t *m) {
	/**/
	__m128 a = _mm_load_ps((float*)x);
	__m128 m0 = _mm_load_ps(m->m[0]);
	__m128 m1 = _mm_load_ps(m->m[1]);
	__m128 m2 = _mm_load_ps(m->m[2]);
	__m128 m3 = _mm_load_ps(m->m[3]);
	__m128 r = _mm_mul_ps(_mm_repx_ps(a), m0);
	r = _mm_madd_ps(_mm_repy_ps(a), m1, r);
	r = _mm_madd_ps(_mm_repz_ps(a), m2, r);
	r = _mm_madd_ps(_mm_repw_ps(a), m3, r);
	_mm_store_ps((float*)y, r);

	/**
	float X = x->x, Y = x->y, Z = x->z, W = x->w;
	y->x = X * m->m[0][0] + Y * m->m[1][0] + Z * m->m[2][0] + W * m->m[3][0];
	y->y = X * m->m[0][1] + Y * m->m[1][1] + Z * m->m[2][1] + W * m->m[3][1];
	y->z = X * m->m[0][2] + Y * m->m[1][2] + Z * m->m[2][2] + W * m->m[3][2];
	y->w = X * m->m[0][3] + Y * m->m[1][3] + Z * m->m[2][3] + W * m->m[3][3];
	/**/
#ifdef CALL_CNT 
	mapp++;
#endif
	return y;
}

float matrix_apply_homogenous(float* y, const float* x, const matrix_t *m) {
	float fw;
	__m128 w;
	__m128 a = _mm_load_ps(x);
	__m128 m0 = _mm_load_ps(m->m[0]);
	__m128 m1 = _mm_load_ps(m->m[1]);
	__m128 m2 = _mm_load_ps(m->m[2]);
	__m128 m3 = _mm_load_ps(m->m[3]);
	__m128 r = _mm_mul_ps(_mm_repx_ps(a), m0);
	r = _mm_madd_ps(_mm_repy_ps(a), m1, r);
	r = _mm_madd_ps(_mm_repz_ps(a), m2, r);
	r = _mm_madd_ps(_mm_repw_ps(a), m3, r);
	w = _mm_repw_ps(r);
	_mm_store_ss(&fw, w);
	r = _mm_div_ps(r, w);
	_mm_store_ps(y, r);
	return 1 / fw;
}

matrix_t* matrix_set_identity(matrix_t *m) {
#ifdef CALL_CNT 
	mid++;
#endif
	m->m[0][0] = m->m[1][1] = m->m[2][2] = m->m[3][3] = 1.f;
	m->m[0][1] = m->m[0][2] = m->m[0][3] = 0.f;
	m->m[1][0] = m->m[1][2] = m->m[1][3] = 0.f;
	m->m[2][0] = m->m[2][1] = m->m[2][3] = 0.f;
	m->m[3][0] = m->m[3][1] = m->m[3][2] = 0.f;
	return m;
}

matrix_t* matrix_set_zero(matrix_t *m) {
#ifdef CALL_CNT 
	mzero++;
#endif
	m->m[0][0] = m->m[0][1] = m->m[0][2] = m->m[0][3] = 0.f;
	m->m[1][0] = m->m[1][1] = m->m[1][2] = m->m[1][3] = 0.f;
	m->m[2][0] = m->m[2][1] = m->m[2][2] = m->m[2][3] = 0.f;
	m->m[3][0] = m->m[3][1] = m->m[3][2] = m->m[3][3] = 0.f;
	return m;
}

// 平移变换
matrix_t* matrix_set_translate(matrix_t *m, float x, float y, float z) {
#ifdef CALL_CNT 
	mtra++;
#endif
	matrix_set_identity(m);
	m->m[3][0] = x;
	m->m[3][1] = y;
	m->m[3][2] = z;
	return m;
}

// 缩放变换
matrix_t* matrix_set_scale(matrix_t *m, float x, float y, float z) {
#ifdef CALL_CNT 
	mssca++;
#endif
	matrix_set_identity(m);
	m->m[0][0] = x;
	m->m[1][1] = y;
	m->m[2][2] = z;
	return m;
}

// 旋转矩阵
matrix_t* matrix_set_rotate(matrix_t *m, float x, float y, float z, float theta) {
	float qsin = (float)sin(theta * .5f);
	float w = (float)cos(theta * .5f);
	vector_t vec = { x, y, z, 1.f };
#ifdef CALL_CNT 
	mrot++;
#endif
	vector_normalize(&vec);
	x = vec.x * qsin;
	y = vec.y * qsin;
	z = vec.z * qsin;
	m->m[0][0] = 1 - 2 * y * y - 2 * z * z;
	m->m[1][0] = 2 * x * y - 2 * w * z;
	m->m[2][0] = 2 * x * z + 2 * w * y;
	m->m[0][1] = 2 * x * y + 2 * w * z;
	m->m[1][1] = 1 - 2 * x * x - 2 * z * z;
	m->m[2][1] = 2 * y * z - 2 * w * x;
	m->m[0][2] = 2 * x * z - 2 * w * y;
	m->m[1][2] = 2 * y * z + 2 * w * x;
	m->m[2][2] = 1 - 2 * x * x - 2 * y * y;
	m->m[0][3] = m->m[1][3] = m->m[2][3] = 0.f;
	m->m[3][0] = m->m[3][1] = m->m[3][2] = 0.f;
	m->m[3][3] = 1.f;
	return m;
}

// 设置摄像机
matrix_t* matrix_set_lookat(matrix_t *m, const vector_t *eye, const vector_t *at, const vector_t *up) {
	vector_t xaxis, yaxis, zaxis;
#ifdef CALL_CNT 
	mlook++;
#endif
	vector_sub(&zaxis, at, eye);
	vector_normalize(&zaxis);
	vector_crossproduct(&xaxis, up, &zaxis);
	vector_normalize(&xaxis);
	vector_crossproduct(&yaxis, &zaxis, &xaxis);

	m->m[0][0] = xaxis.x;
	m->m[1][0] = xaxis.y;
	m->m[2][0] = xaxis.z;
	m->m[3][0] = -vector_dotproduct(&xaxis, eye);

	m->m[0][1] = yaxis.x;
	m->m[1][1] = yaxis.y;
	m->m[2][1] = yaxis.z;
	m->m[3][1] = -vector_dotproduct(&yaxis, eye);

	m->m[0][2] = zaxis.x;
	m->m[1][2] = zaxis.y;
	m->m[2][2] = zaxis.z;
	m->m[3][2] = -vector_dotproduct(&zaxis, eye);

	m->m[0][3] = m->m[1][3] = m->m[2][3] = 0.f;
	m->m[3][3] = 1.f;
	return m;
}

// D3DXMatrixPerspectiveFovLH
matrix_t* matrix_set_perspective(matrix_t *m, float fovy, float aspect, float zn, float zf) {
	float fax = 1.f / (float)tan(fovy * .5f);
#ifdef CALL_CNT 
	mproj++;
#endif
	matrix_set_zero(m);
	m->m[0][0] = (float)(fax / aspect);
	m->m[1][1] = (float)(fax);
	m->m[2][2] = zf / (zf - zn);
	m->m[3][2] = -zn * zf / (zf - zn);
	m->m[2][3] = 1;
	return m;
}

typedef struct {
	matrix_t world;         // 世界坐标变换
	matrix_t view;          // 摄影机坐标变换
	matrix_t projection;    // 投影变换
	matrix_t light;			// 世界-光源空间变换
	matrix_t o2l;			// 物体-光源空间变换
	matrix_t transform;     // transform = world * view * projection
	float w, h;             // 屏幕大小
}	transform_t;

// 矩阵更新，计算 transform = world * view * projection
void transform_update(transform_t *ts) {
	matrix_t m;
	matrix_mul(&m, &ts->world, &ts->view);
	matrix_mul(&ts->transform, &m, &ts->projection);
	matrix_mul(&ts->o2l, &ts->world, &ts->light);
}

// 初始化，设置屏幕长宽
void transform_init(transform_t *ts, int width, int height) {
	float aspect = (float)width / ((float)height);
	matrix_set_identity(&ts->world);
	matrix_set_identity(&ts->view);
	matrix_set_perspective(&ts->projection, 3.1415926f * .3278f, aspect, 1.f, 500.f);
	ts->w = (float)width;
	ts->h = (float)height;
	transform_update(ts);
}

// 检查齐次坐标同 cvv 的边界用于视锥裁剪
int transform_check_cvv(const float *v) {
	int check = 0;
	if (v[2] < 0.f) check |= 1;
	if (v[2] > 1.f) check |= 2;
	if (v[0] < -1.f) check |= 4;
	if (v[0] > 1.f) check |= 8;
	if (v[1] < -1.f) check |= 16;
	if (v[1] > 1.f) check |= 32;
	return check;
}

// 转换到屏幕坐标系
void transform_to_screen(const transform_t* ts, float* y, const float* x) {
	y[0] = (x[0] + 1.f) * ts->w * .5f;
	y[1] = (1.f - x[1]) * ts->h * .5f;
	y[3] = 1.f;
}

typedef struct { float r, g, b; } color_t;
typedef __declspec(align(16)) struct { float u, v; } texcoord_t;

#define VL 20
int vl = VL, dst_size = sizeof(float) * VL, src_size = sizeof(float) * VL;
typedef struct { point_t pos; vector_t normal; texcoord_t tc; color_t color; } vertex_t;
typedef struct { __declspec(align(16)) float v[VL], v1[VL], v2[VL]; } edge_t;
typedef struct { float top, bottom; edge_t left, right; } trapezoid_t;
typedef struct { __declspec(align(16)) float v[VL], step[VL]; int x, y, w; } scanline_t;

void vertex_interp(float* y, const float* x1, const float* x2, float t) {
	__m128 v1, v2, v, p;
	int i;
	p = _mm_repx_ps(_mm_load_ss(&t));
	for (i = 0; i < vl; i += 4) {
		v1 = _mm_load_ps(&x1[i]);
		v2 = _mm_load_ps(&x2[i]);
		v = _mm_add_ps(v1, _mm_mul_ps(p, _mm_sub_ps(v2, v1)));
		_mm_store_ps(&y[i], v);
	}
}

void vertex_division(float* y, const float* x1, const float* x2, float w) {
	float inv = 1.f / w;
	__m128 v1, v2, v, p;
	int i;
	p = _mm_repx_ps(_mm_load_ss(&inv));
	for (i = 0; i < vl; i += 4) {
		v1 = _mm_load_ps(&x1[i]);
		v2 = _mm_load_ps(&x2[i]);
		v = _mm_mul_ps(p, _mm_sub_ps(v2, v1));
		_mm_store_ps(&y[i], v);
	}
}

void vertex_add(float* y, const float* x) {
	__m128 v1, v2, v;
	int i;
	for (i = 0; i < vl; i += 4) {
		v1 = _mm_load_ps(&x[i]);
		v2 = _mm_load_ps(&y[i]);
		v = _mm_add_ps(v1, v2);
		_mm_store_ps(&y[i], v);
	}
}

// 根据三角形生成 0-2 个梯形，并且返回合法梯形的数量
int trapezoid_init_triangle(trapezoid_t *trap, const float* p1,
	const float* p2, const float* p3) {
	float k, x;
	const float* p;

	if (p1[1] > p2[1]) p = p1, p1 = p2, p2 = p;
	if (p1[1] > p3[1]) p = p1, p1 = p3, p3 = p;
	if (p2[1] > p3[1]) p = p2, p2 = p3, p3 = p;
	if (p1[1] == p2[1] && p1[1] == p3[1]) return 0;
	if (p1[0] == p2[0] && p1[0] == p3[0]) return 0;

	if (p1[1] == p2[1]) { // down
		if (p1[0] > p2[0]) p = p1, p1 = p2, p2 = p;
		trap[0].top = p1[1];
		trap[0].bottom = p3[1];
		memcpy_s(trap[0].left.v1, dst_size, p1, src_size);
		memcpy_s(trap[0].left.v2, dst_size, p3, src_size);
		memcpy_s(trap[0].right.v1, dst_size, p2, src_size);
		memcpy_s(trap[0].right.v2, dst_size, p3, src_size);
		return (trap[0].top < trap[0].bottom) ? 1 : 0;
	}

	if (p2[1] == p3[1]) { // up
		if (p2[0] > p3[0]) p = p2, p2 = p3, p3 = p;
		trap[0].top = p1[1];
		trap[0].bottom = p3[1];
		memcpy_s(trap[0].left.v1, dst_size, p1, src_size);
		memcpy_s(trap[0].left.v2, dst_size, p2, src_size);
		memcpy_s(trap[0].right.v1, dst_size, p1, src_size);
		memcpy_s(trap[0].right.v2, dst_size, p3, src_size);
		return (trap[0].top < trap[0].bottom) ? 1 : 0;
	}

	trap[0].top = p1[1];
	trap[0].bottom = p2[1];
	trap[1].top = p2[1];
	trap[1].bottom = p3[1];

	k = (p3[1] - p1[1]) / (p2[1] - p1[1]);
	x = p1[0] + (p2[0] - p1[0]) * k;

	if (x <= p3[0]) { // left
		memcpy_s(trap[0].left.v1, dst_size, p1, src_size);
		memcpy_s(trap[0].left.v2, dst_size, p2, src_size);
		memcpy_s(trap[0].right.v1, dst_size, p1, src_size);
		memcpy_s(trap[0].right.v2, dst_size, p3, src_size);
		memcpy_s(trap[1].left.v1, dst_size, p2, src_size);
		memcpy_s(trap[1].left.v2, dst_size, p3, src_size);
		memcpy_s(trap[1].right.v1, dst_size, p1, src_size);
		memcpy_s(trap[1].right.v2, dst_size, p3, src_size);
	}
	else { // right
		memcpy_s(trap[0].left.v1, dst_size, p1, src_size);
		memcpy_s(trap[0].left.v2, dst_size, p3, src_size);
		memcpy_s(trap[0].right.v1, dst_size, p1, src_size);
		memcpy_s(trap[0].right.v2, dst_size, p2, src_size);
		memcpy_s(trap[1].left.v1, dst_size, p1, src_size);
		memcpy_s(trap[1].left.v2, dst_size, p3, src_size);
		memcpy_s(trap[1].right.v1, dst_size, p2, src_size);
		memcpy_s(trap[1].right.v2, dst_size, p3, src_size);
	}

	return 2;
}

// 按照 Y 坐标计算出左右两条边纵坐标等于 Y 的顶点
void trapezoid_edge_interp(trapezoid_t *trap, float y) {
	float s1 = trap->left.v2[1] - trap->left.v1[1];
	float s2 = trap->right.v2[1] - trap->right.v1[1];
	float t1 = (y - trap->left.v1[1]) / s1;
	float t2 = (y - trap->right.v1[1]) / s2;
	vertex_interp(trap->left.v, trap->left.v1, trap->left.v2, t1);
	vertex_interp(trap->right.v, trap->right.v1, trap->right.v2, t2);
}

// 根据左右两边的端点，初始化计算出扫描线的起点和步长
void trapezoid_init_scan_line(const trapezoid_t *trap, scanline_t *scanline, int y) {
	float width = trap->right.v[0] - trap->left.v[0];
	scanline->x = (int)(trap->left.v[0] + .5f);
	scanline->w = (int)(trap->right.v[0] + .5f) - scanline->x;
	scanline->y = y;
	memcpy_s(scanline->v, dst_size, trap->left.v, src_size);
	if (trap->left.v[0] >= trap->right.v[0]) scanline->w = 0;
	vertex_division(scanline->step, trap->left.v, trap->right.v, width);
}

typedef struct {
	transform_t transform;      // 坐标变换器
	int width;                  // 窗口宽度
	int height;                 // 窗口高度
	IUINT32 **framebuffer;      // 像素缓存：framebuffer[y] 代表第 y行
	float **zbuffer;            // 深度缓存：zbuffer[y] 为第 y行指针
	IUINT32 **texture;          // 纹理：同样是每行索引
	int tex_width;              // 纹理宽度
	int tex_height;             // 纹理高度
	float max_u;                // 纹理最大宽度：tex_width - 1
	float max_v;                // 纹理最大高度：tex_height - 1
	int render_state;           // 渲染状态
	IUINT32 background;         // 背景颜色
	IUINT32 foreground;         // 线框颜色
	vector_t light_dir;			// 平行光方向
	__m128 ambient;				// 环境光颜色
}	device_t;

#define RENDER_STATE_POINT	        1		// 渲染点阵
#define RENDER_STATE_WIREFRAME      2		// 渲染线框
#define RENDER_STATE_COLOR	        4		// 渲染颜色
#define RENDER_STATE_TEXTURE        8		// 渲染纹理

char shadowmap_test(device_t* device, const point_t* p);

void update_light_matrix(device_t* device) {
	vector_t eye = { -4.f, -4.f, 0.f, 1.f }, at, up = { 0.f, 1.f, 0.f, 0.f };
	vector_add(&at, &device->light_dir, &eye);
	matrix_set_lookat(&device->transform.light, &eye, &at, &up);
}

// 设备初始化，fb为外部帧缓存，非 NULL 将引用外部帧缓存（每行 4字节对齐）
void device_init(device_t *device, int width, int height, void *fb) {
	int need = sizeof(void*) * (height * 2 + 1024) + width * height * 8;
	char *ptr = (char*)malloc(need + 64);
	char *framebuf, *zbuf;
	int j;
	vector_t ambient = { .15f, .15f, .15f, 0.f };
	assert(ptr);
	device->framebuffer = (IUINT32**)ptr;
	device->zbuffer = (float**)(ptr + sizeof(void*) * height);
	ptr += sizeof(void*) * height * 2;
	device->texture = (IUINT32**)ptr;
	ptr += sizeof(void*) * 1024;
	framebuf = (char*)ptr;
	zbuf = (char*)ptr + width * height * 4;
	ptr += width * height * 8;
	if (fb != NULL)
		framebuf = (char*)fb;
	for (j = 0; j < height; j++) {
		device->framebuffer[j] = (IUINT32*)(framebuf + width * 4 * j);
		device->zbuffer[j] = (float*)(zbuf + width * 4 * j);
	}
	device->texture[0] = (IUINT32*)ptr;
	device->texture[1] = (IUINT32*)(ptr + 16);
	memset(device->texture[0], 0, 64);
	device->tex_width = 2;
	device->tex_height = 2;
	device->max_u = 1.f;
	device->max_v = 1.f;
	device->width = width;
	device->height = height;
	device->background = 0xc0c0c0;//0xfffec22a;
	device->foreground = 0;
	device->light_dir.x = 0.f;
	device->light_dir.y = 0.f;
	device->light_dir.z = 1.f;
	device->light_dir.w = 0.f;
	device->ambient = _mm_load_ps((float*)&ambient);
	transform_init(&device->transform, width, height);
	update_light_matrix(device);
	device->render_state = RENDER_STATE_WIREFRAME;
}

// 删除设备
void device_destroy(device_t *device) {
	if (device->framebuffer)
		free(device->framebuffer);
	device->framebuffer = NULL;
	device->zbuffer = NULL;
	device->texture = NULL;
}

// 设置当前纹理
void device_set_texture(device_t *device, void *bits, long pitch, int w, int h) {
	char *ptr = (char*)bits;
	int j;
	assert(w <= 1024 && h <= 1024);
	for (j = 0; j < h; ptr += pitch, j++) 	// 重新计算每行纹理的指针
		device->texture[j] = (IUINT32*)ptr;
	device->tex_width = w;
	device->tex_height = h;
	device->max_u = (float)(w - 1);
	device->max_v = (float)(h - 1);
}

// 清空 framebuffer 和 zbuffer
void device_clear(device_t *device) {
	int w = device->width, h = device->height;
	IUINT32 cc = device->background;
	memset(device->framebuffer[0], cc, sizeof(IUINT32) * w * h);
	memset(device->zbuffer[0], 0, sizeof(float) * w * h);
}

// 画点
void device_pixel(device_t *device, int x, int y, IUINT32 color, float rhw) {
	if (((IUINT32)x) < (IUINT32)device->width && ((IUINT32)y) < (IUINT32)device->height) {
		if (rhw >= device->zbuffer[y][x]) {
			device->framebuffer[y][x] = color;
			device->zbuffer[y][x] = rhw;
		}
	}
}

// 绘制线段
void device_draw_line(device_t *device, const float* v1, const float* v2, IUINT32 c) {
	int x1 = (int)v1[0], x2 = (int)v2[0], y1 = (int)v1[1], y2 = (int)v2[1];
	float rhw = v1[vl - 1], rhwstp = (v2[vl - 1] - v1[vl - 1]);
	int x, y, rem = 0, offset;
	if (x1 == x2 && y1 == y2) {
		device_pixel(device, x1, y1, c, v1[vl - 1]);
	}
	else if (x1 == x2) {
		int inc = (y1 <= y2) ? 1 : -1;
		rhwstp /= abs(y1 - y2);
		for (y = y1; y != y2; y += inc, rhw += rhwstp) device_pixel(device, x1, y, c, rhw);
		device_pixel(device, x2, y2, c, rhw);
	}
	else if (y1 == y2) {
		int inc = (x1 <= x2) ? 1 : -1;
		rhwstp /= abs(x1 - x2);
		for (x = x1; x != x2; x += inc, rhw += rhwstp) device_pixel(device, x, y1, c, rhw);
		device_pixel(device, x2, y2, c, rhw);
	}
	else { // Bresenham
		int dx = abs(x2 - x1);
		int dy = abs(y2 - y1);
		if (dx >= dy) {
			rhwstp /= dx;
			if (x2 < x1) x = x1, y = y1, x1 = x2, y1 = y2, x2 = x, y2 = y, rhw = v2[vl - 1], rhwstp = -rhwstp;
			offset = (y2 >= y1) ? 1 : -1;
			for (x = x1, y = y1; x <= x2; x++, rhw += rhwstp) {
				device_pixel(device, x, y, c, rhw);
				rem += dy;
				if (rem >= dx) {
					rem -= dx;
					y += offset;
					device_pixel(device, x, y, c, rhw);
				}
			}
			device_pixel(device, x2, y2, c, rhw);
		}
		else {
			rhwstp /= dy;
			if (y2 < y1) x = x1, y = y1, x1 = x2, y1 = y2, x2 = x, y2 = y, rhw = v2[vl - 1], rhwstp = -rhwstp;
			offset = (x2 > x1) ? 1 : -1;
			for (x = x1, y = y1; y <= y2; y++, rhw += rhwstp) {
				device_pixel(device, x, y, c, rhw);
				rem += dx;
				if (rem >= dy) {
					rem -= dy;
					x += offset;
					device_pixel(device, x, y, c, rhw);
				}
			}
			device_pixel(device, x2, y2, c, rhw);
		}
	}
}

// 根据坐标读取纹理
void device_texture_read(color_t *cc, const device_t *device, float u, float v) {
	/**
	IUINT32 c = device->texture[(int)(v * (device->max_v - 1))][(int)(u * (device->max_u - 1))];
	cc->r = (float) ((c & 0xff0000) >> 16);
	cc->g = (float) ((c & 0xff00) >> 8);
	cc->b = (float) (c & 0xff);
	/**/
	int x, y; float dx, dy, wx, wy;
	IUINT32 c00, c01, c10, c11;
	IUINT32 r00, r01, r10, r11, g00, g01, g10, g11, b00, b01, b10, b11;
	u = u * (device->max_u - 1);
	v = v * (device->max_v - 1);
	x = (int)u;
	y = (int)v;
	//x = CMID(x, 0, device->tex_width - 2); // Clamp
	//y = CMID(y, 0, device->tex_height - 2);
	// Bilinear Filtering
	dx = (u - x); dy = (v - y); wx = 1 - dx; wy = 1 - dy;
	c00 = device->texture[y][x]; c01 = device->texture[y][x + 1];
	c10 = device->texture[y + 1][x]; c11 = device->texture[y + 1][x + 1];
	r00 = (c00 & 0xff0000) >> 16; g00 = (c00 & 0xff00) >> 8; b00 = c00 & 0xff;
	r01 = (c01 & 0xff0000) >> 16; g01 = (c01 & 0xff00) >> 8; b01 = c01 & 0xff;
	r10 = (c10 & 0xff0000) >> 16; g10 = (c10 & 0xff00) >> 8; b10 = c10 & 0xff;
	r11 = (c11 & 0xff0000) >> 16; g11 = (c11 & 0xff00) >> 8; b11 = c11 & 0xff;
	cc->r = r00*wx*wy + r01*dx*wy + r10*wx*dy + r11*dx*dy;
	cc->g = g00*wx*wy + g01*dx*wy + g10*wx*dy + g11*dx*dy;
	cc->b = b00*wx*wy + b01*dx*wy + b10*wx*dy + b11*dx*dy;
	/**/
}

void colorPS(device_t* device, color_t* o, scanline_t* sl, IUINT32* fb, float* zb, int x) {
	float rhw = sl->v[vl - 1], *v, w;
	int R, G, B;
	if (rhw < *zb) return;
	v = sl->v;
	w = 1 / rhw;
	*zb = rhw;
	o->r = v[4] * w * 255;
	o->g = v[5] * w * 255;
	o->b = v[6] * w * 255;
	R = CMID((int)o->r, 0, 255);
	G = CMID((int)o->g, 0, 255);
	B = CMID((int)o->b, 0, 255);
	*fb = (R << 16) | (G << 8) | (B);
}
void texturePS(device_t* device, color_t* o, scanline_t* sl, IUINT32* fb, float* zb, int x) {
	float rhw = sl->v[vl - 1], *v, w, tu, tv, shadow, specular;
	int R, G, B;
	if (rhw < *zb) return;
	v = sl->v;
	w = 1 / rhw;
	tu = v[16] * w;
	tv = v[17] * w;
	shadow = shadowmap_test(device, (point_t*)&v[8]) ? .1f : 1.f;
	specular = v[15] * w;
	*zb = rhw;
	device_texture_read(o, device, tu, tv);
	o->r = (o->r * v[12] * w + specular) * shadow;
	o->g = (o->g * v[13] * w + specular) * shadow;
	o->b = (o->b * v[14] * w + specular) * shadow;
	R = CMID((int)o->r, 0, 255);
	G = CMID((int)o->g, 0, 255);
	B = CMID((int)o->b, 0, 255);
	*fb = (R << 16) | (G << 8) | (B);
}
void(*PS_func)(device_t* device, color_t* o, scanline_t* sl, IUINT32* fb, float* zb, int x) = colorPS;
// 绘制扫描线
void device_draw_scanline(device_t *device, scanline_t* scanline) {
	IUINT32 *framebuffer = device->framebuffer[scanline->y];
	float *zbuffer = device->zbuffer[scanline->y];
	int x = scanline->x;
	int w = scanline->w;
	int width = device->width;
	color_t cc;
	float cDiffuse = 1.f;
	for (; w > 0; x++, w--) {
		if (x >= 0 && x < width)
			PS_func(device, &cc, scanline, &framebuffer[x], &zbuffer[x], x);
		vertex_add(scanline->v, scanline->step);
		if (x >= width) break;
	}
}

// 主渲染函数
void device_render_trap(device_t *device, trapezoid_t *trap) {
	scanline_t scanline;
	int j, top, bottom;
	top = (int)(trap->top + .5f);
	bottom = (int)(trap->bottom + .5f);
	for (j = top; j < bottom; j++) {
		if (j >= 0 && j < device->height) {
			trapezoid_edge_interp(trap, (float)j + .5f);
			trapezoid_init_scan_line(trap, &scanline, j);
			device_draw_scanline(device, &scanline);
		}
		if (j >= device->height) break;
	}
}

void wireframeVS(device_t* device, float* o, const vertex_t *v) {}
void colorVS(device_t* device, float* o, const vertex_t *v) {
	__m128 c, w;
	c = _mm_load_ps((float*)&v->color);
	w = _mm_repx_ps(_mm_load_ss(&o[vl - 1]));
	c = _mm_mul_ps(c, w);
	_mm_store_ps(&o[4], c);
	_mm_store_ss(&o[vl - 1], w);
}

float shadowmap[1024 * 1024];
int sm_res = 1024;
int sm_len = 1024 * 1024;
void textureVS(device_t* device, float* o, const vertex_t *v) {
	__m128 t, c, w;
	vector_t Hw;
	matrix_apply((vector_t*)&o[4], &v->normal, &device->transform.world);
	matrix_apply((vector_t*)&o[8], &v->pos, &device->transform.o2l);

	o[8] = o[8] * .083f * sm_res;
	o[9] = o[9] * .083f * sm_res;
	vector_normalize((vector_t*)&o[4]);
	o[12] = o[13] = o[14] = saturate(-vector_dotproduct(&device->light_dir, (vector_t*)&o[4]));

	vector_add(&Hw, (vector_t*)&o[4], &device->light_dir);
	vector_normalize(&Hw);
	o[15] = saturate(powf(vector_dotproduct((vector_t*)&o[4], &Hw), 400)) * 255;

	t = _mm_load_ps((float*)&v->tc);
	c = _mm_add_ps(device->ambient, _mm_load_ps(&o[12]));
	w = _mm_repx_ps(_mm_load_ss(&o[vl - 1]));
	t = _mm_mul_ps(t, w);
	c = _mm_mul_ps(c, w);
	_mm_store_ps(&o[12], c);
	_mm_store_ps(&o[16], t);
	_mm_store_ss(&o[vl - 1], w);
}
void(*VS_func)(device_t* ts, float* y, const vertex_t *x) = textureVS;

int near_plane_clip(float v[5][VL], int* id, int* c) {
	// 完全在平截头体内
	if (!c[0] && !c[1] && !c[2]) return 3;

	// 近平面裁剪
	if ((c[0] | c[1] | c[2]) & 1) {
		int out[2] = { 0, 1 }, cnt = 0;
		if (c[0] & 1) cnt++;
		if (c[1] & 1) out[cnt++] = 1;
		if (c[2] & 1) out[cnt++] = 2;
		if (cnt == 1) {
			id[0] = (out[0] + 1) % 3;
			id[1] = (out[0] + 2) % 3;
			id[2] = 3;
			id[3] = 4;
			vertex_interp(v[3], v[id[1]], v[out[0]], v[id[1]][2] / (v[id[1]][2] - v[out[0]][2]));
			vertex_interp(v[4], v[id[0]], v[out[0]], v[id[0]][2] / (v[id[0]][2] - v[out[0]][2]));
			return 4;
		}
		else {
			id[0] = 3 - out[0] - out[1];
			id[1] = 3;
			id[2] = 4;
			vertex_interp(v[3], v[id[0]], v[out[0]], v[id[0]][2] / (v[id[0]][2] - v[out[0]][2]));
			vertex_interp(v[4], v[id[0]], v[out[1]], v[id[0]][2] / (v[id[0]][2] - v[out[1]][2]));
			return 3;
		}
	}

	return 3;
}

void wvp_transform(const transform_t* ts, float v[VL], const float* p) {
	v[vl - 1] = matrix_apply_homogenous(v, p, &ts->transform);
}
void(*transform_func)(const transform_t* ts, float v[VL], const float* p) = wvp_transform;

char ccw_cull(float v[5][VL], int* c) {
	vector_t a1, a2;
	vector_sub(&a1, (vector_t*)v[1], (vector_t*)v[0]);
	vector_sub(&a2, (vector_t*)v[2], (vector_t*)v[0]);
	if (a1.x*a2.y - a2.x*a1.y > 0) return 1;
	return 0;
}
char frustum_cull(float v[5][VL], int* c) {
	c[0] = transform_check_cvv(v[0]);
	c[1] = transform_check_cvv(v[1]);
	c[2] = transform_check_cvv(v[2]);

	if (v[0][vl - 1] < 0.f || v[1][vl - 1] < 0.f || v[2][vl - 1] < 0.f)
		return 1;
	// 完全在平截头体外
	if (c[0] & c[1] & c[2]) return 1;
	// CCW背面剔除
	return ccw_cull(v, c);
}
char(*cull_func)(float v[5][VL], int* c) = frustum_cull;

int RS_CLIPPING = 1;
int RS_SCREEN_MAPPING = 1;
// pipeline
void device_draw_primitive(device_t *device, const vertex_t *v1,
	const vertex_t *v2, const vertex_t *v3) {
	static __declspec(align(16)) float t[5][VL];
	static int id[5];
	int render_state = device->render_state;
	int vertex_cnt, i, c[3];
	id[0] = 0; id[1] = 1; id[2] = 2;

	transform_func(&device->transform, t[0], &v1->pos.x);
	transform_func(&device->transform, t[1], &v2->pos.x);
	transform_func(&device->transform, t[2], &v3->pos.x);

	// 剔除
	if (cull_func(t, c)) return;

	VS_func(device, t[0], v1);
	VS_func(device, t[1], v2);
	VS_func(device, t[2], v3);

	// 裁剪
	vertex_cnt = RS_CLIPPING ? near_plane_clip(t, id, c) : 3;

	//屏幕映射
	if (RS_SCREEN_MAPPING) {
		for (i = 0; i < vertex_cnt; i++)
			transform_to_screen(&device->transform, t[id[i]], t[id[i]]);
	}

	if (render_state & RENDER_STATE_POINT) { // 点绘制
		for (i = 0; i < vertex_cnt; i++)
			device_pixel(device, (int)t[id[i]][0], (int)t[id[i]][1], device->foreground, t[id[i]][vl - 1]);
	}
	else if (render_state & RENDER_STATE_WIREFRAME) { // 线框绘制
		for (i = 0; i < vertex_cnt; i++)
			device_draw_line(device, t[id[i]], t[id[(i + 1) % vertex_cnt]], device->foreground);
	}
	else { // 纹理或者色彩绘制
		trapezoid_t traps[2];
		int n;
		for (i = 2; i < vertex_cnt; i++) {
			n = trapezoid_init_triangle(traps, t[id[0]], t[id[i - 1]], t[id[i]]);
			if (n >= 1) device_render_trap(device, &traps[0]);
			if (n >= 2) device_render_trap(device, &traps[1]);
		}
	}
}

void device_set_render_state(device_t* device, int rs) {
	switch (rs) {
	case RENDER_STATE_POINT:
	case RENDER_STATE_WIREFRAME:
		VS_func = wireframeVS;
		vl = 4;
		break;
	case RENDER_STATE_COLOR:
		VS_func = colorVS;
		PS_func = colorPS;
		vl = 8;
		break;
	case RENDER_STATE_TEXTURE:
		VS_func = textureVS;
		PS_func = texturePS;
		vl = 20;
		break;
	}
	transform_func = wvp_transform;
	cull_func = frustum_cull;
	RS_CLIPPING = 1;
	RS_SCREEN_MAPPING = 1;
	src_size = sizeof(float) * vl;
	device->render_state = rs;

}

int screen_w, screen_h, screen_exit = 0;
int screen_mx = 0, screen_my = 0, screen_mb = 0;
short mouse_x, mouse_y, drag_x, drag_y; // 当前/起始鼠标位置
char screen_keys[256];					// 当前键鼠按下状态
static HWND screen_handle = NULL;		// 主窗口 HWND
static HDC screen_dc = NULL;			// 配套的 HDC
static HBITMAP screen_hb = NULL;		// DIB
static HBITMAP screen_ob = NULL;		// 老的 BITMAP
static HFONT screen_font = NULL;		// FPS显示字体
static RECT screen_rect = { 0, 0, 0, 0 }; // 字体显示区域
unsigned char *screen_fb = NULL;		// frame buffer

int screen_init(int w, int h, const TCHAR *title);	// 屏幕初始化
int screen_close();									// 关闭屏幕
void screen_dispatch();								// 处理消息
void screen_update();								// 显示 FrameBuffer

													// win32 event handler
static LRESULT screen_events(HWND, UINT, WPARAM, LPARAM);

// 初始化窗口并设置标题
int screen_init(int w, int h, const TCHAR *title) {
	WNDCLASS wc = { CS_BYTEALIGNCLIENT, (WNDPROC)screen_events, 0, 0, 0,
		NULL, NULL, NULL, NULL, _T("Mini3d") };
	BITMAPINFO bi = { { sizeof(BITMAPINFOHEADER), w, -h, 1, 32, BI_RGB,
		w * h * 4, 0, 0, 0, 0 } };
	RECT rect = { 0, 0, w, h };
	int wx, wy, sx, sy;
	LPVOID ptr;
	HDC hDC;

	screen_close();

	wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wc.hInstance = GetModuleHandle(NULL);
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);
	wc.hIcon = (HICON)LoadImage(NULL, MAKEINTRESOURCE(32516), IMAGE_ICON,
		0, 0, LR_DEFAULTSIZE | LR_SHARED);
	if (!RegisterClass(&wc)) return -1;

	screen_handle = CreateWindow(wc.lpszClassName, title,
		WS_SYSMENU | WS_MINIMIZEBOX,
		0, 0, 0, 0, NULL, NULL, wc.hInstance, NULL);
	if (screen_handle == NULL) return -2;

	screen_exit = 0;
	hDC = GetDC(screen_handle);
	screen_dc = CreateCompatibleDC(hDC);
	ReleaseDC(screen_handle, hDC);

	screen_hb = CreateDIBSection(screen_dc, &bi, DIB_RGB_COLORS, &ptr, 0, 0);
	if (screen_hb == NULL) return -3;

	screen_ob = (HBITMAP)SelectObject(screen_dc, screen_hb);
	screen_fb = (unsigned char*)ptr;
	screen_w = w;
	screen_rect.right = w - 5;
	screen_rect.bottom = screen_h = h;

	screen_font = CreateFont(30, 0, 0, 0, 0, 0, 0, 0, ANSI_CHARSET, 0, 0,
		ANTIALIASED_QUALITY, 0, _T("Consolas"));
	SetTextColor(screen_dc, RGB(0, 0, 0));
	SetBkMode(screen_dc, TRANSPARENT);

	AdjustWindowRect(&rect, (DWORD)GetWindowLongPtr(screen_handle, GWL_STYLE), 0);
	wx = rect.right - rect.left;
	wy = rect.bottom - rect.top;
	sx = (GetSystemMetrics(SM_CXSCREEN) - wx) / 2;
	sy = (GetSystemMetrics(SM_CYSCREEN) - wy) / 2;
	if (sy < 0) sy = 0;
	SetWindowPos(screen_handle, NULL, sx, sy, wx, wy, (SWP_NOCOPYBITS | SWP_NOZORDER | SWP_SHOWWINDOW));
	SetForegroundWindow(screen_handle);

	screen_dispatch();

	memset(screen_keys, 0, sizeof(char) * 256);
	memset(screen_fb, 0, w * h * 4);

	return 0;
}

int screen_close() {
	if (screen_dc) {
		if (screen_ob) {
			SelectObject(screen_dc, screen_ob);
			screen_ob = NULL;
		}
		DeleteDC(screen_dc);
		screen_dc = NULL;
	}
	if (screen_hb) {
		DeleteObject(screen_hb);
		screen_hb = NULL;
	}
	if (screen_handle) {
		CloseWindow(screen_handle);
		screen_handle = NULL;
	}
	return 0;
}

static LRESULT screen_events(HWND hWnd, UINT msg,
	WPARAM wParam, LPARAM lParam) {
	switch (msg) {
	case WM_CLOSE: screen_exit = 1; break;
	case WM_KEYDOWN: screen_keys[wParam] = 1; break;
	case WM_KEYUP: screen_keys[wParam] = 0; break;
	case WM_LBUTTONDOWN: screen_keys[VK_LBUTTON] = 1; break;
	case WM_LBUTTONUP: screen_keys[VK_LBUTTON] = 0; break;
	case WM_RBUTTONDOWN: screen_keys[VK_RBUTTON] = 1; break;
	case WM_RBUTTONUP: screen_keys[VK_RBUTTON] = 0; break;
	case WM_MOUSEWHEEL: screen_keys[VK_MBUTTON] = (char)(GET_WHEEL_DELTA_WPARAM(wParam) / WHEEL_DELTA);
	case WM_MOUSEMOVE: mouse_x = LOWORD(lParam), mouse_y = HIWORD(lParam); break;
	default: return DefWindowProc(hWnd, msg, wParam, lParam);
	}
	return 0;
}

void screen_dispatch() {
	MSG msg;
	while (1) {
		if (!PeekMessage(&msg, NULL, 0, 0, PM_NOREMOVE)) return;
		if (!GetMessage(&msg, NULL, 0, 0)) return;
		DispatchMessage(&msg);
	}
}

void screen_update() {
	HDC hDC = GetDC(screen_handle);
	BitBlt(hDC, 0, 0, screen_w, screen_h, screen_dc, 0, 0, SRCCOPY);
	ReleaseDC(screen_handle, hDC);
	screen_dispatch();
}

vertex_t grid[81];
vertex_t knot[23232];
vertex_t mesh[8] = {
	{ { -1.f, -1.f, -1.f, 1.f },{ -1.f, -1.f, -1.f, 0.f },{ 0.f, 0.f },{ 1.f, .2f, .2f } },
	{ { -1.f, 1.f, -1.f, 1.f },{ -1.f, 1.f, -1.f, 0.f },{ 0.f, 1.f },{ .2f, 1.f, .2f } },
	{ { 1.f, -1.f, -1.f, 1.f },{ 1.f, -1.f, -1.f, 0.f },{ 1.f, 1.f },{ .2f, .2f, 1.f } },
	{ { 1.f, 1.f, -1.f, 1.f },{ 1.f, 1.f, -1.f, 0.f },{ 1.f, 0.f },{ 1.f, .2f, 1.f } },
	{ { -1.f, -1.f, 1.f, 1.f },{ -1.f, -1.f, 1.f, 0.f },{ 0.f, 0.f },{ 1.f, 1.f, .2f } },
	{ { -1.f, 1.f, 1.f, 1.f },{ -1.f, 1.f, 1.f, 0.f },{ 0.f, 1.f },{ .2f, 1.f, 1.f } },
	{ { 1.f, -1.f, 1.f, 1.f },{ 1.f, -1.f, 1.f, 0.f },{ 1.f, 1.f },{ 1.f, .3f, .3f } },
	{ { 1.f, 1.f, 1.f, 1.f },{ 1.f, 1.f, 1.f, 0.f },{ 1.f, 0.f },{ .2f, 1.f, .3f } },
};
unsigned int knot_index[139392];
char states[] = { RENDER_STATE_POINT, RENDER_STATE_WIREFRAME, RENDER_STATE_COLOR, RENDER_STATE_TEXTURE };

void init_texture(device_t *device) {
	static IUINT32 texture[256][256];
	int i, j;
	for (j = 0; j < 256; j++) {
		for (i = 0; i < 256; i++) {
			int x = i / 32, y = j / 32;
			texture[j][i] = ((x + y) & 1) ? 0xffffff : 0;//0x3fbcef;
		}
	}
	device_set_texture(device, texture, 256 * 4, 256, 256);
}

void init_grid() {
	int i, j;
	memset(&grid, 0, sizeof(vertex_t) * 81);
	for (i = 0; i < 9; i++) {
		for (j = 0; j < 9; j++) {
			grid[i * 9 + j].pos.x = j - 4.f;
			grid[i * 9 + j].pos.y = i - 4.f;
			grid[i * 9 + j].pos.z = 20.f;
			grid[i * 9 + j].pos.w = 1.f;
			grid[i * 9 + j].normal.z = -1.f;
			grid[i * 9 + j].tc.u = .9814f;
			grid[i * 9 + j].tc.v = .0185f;
			grid[i * 9 + j].color.r = 1.f;
			grid[i * 9 + j].color.g = 1.f;
			grid[i * 9 + j].color.b = 1.f;
		}
	}
}

void init_knot() {
	FILE *file;
	int i, cnt = 0; float max = -2.f, min = 2.f, sum, scale;
	vector_t *p1, *p2, *p3, v1, v2;
	memset(&knot, 0, sizeof(vertex_t) * 23232);
	memset(&knot_index, 0, sizeof(int) * 139392);
	if (fopen_s(&file, "..\\knot.x", "r") && fopen_s(&file, "knot.x", "r")) {
		MessageBox(0, _T("\"knot.x\" not found"), _T("Missing Model"), MB_OK);
		screen_keys[VK_XBUTTON1] = 1;
		return;
	}
	fscanf_s(file, " %d\n", &cnt);
	for (i = 0; i < cnt; i++) {
		fscanf_s(file, " %f %f %f\n", &knot[i].pos.x, &knot[i].pos.y, &knot[i].pos.z);
		knot[i].pos.x -= .5f; knot[i].pos.y -= .5f; knot[i].pos.z -= .4f;
		knot[i].pos.w = 1.f;
		//knot[i].tc.u = .9814f;
		//knot[i].tc.v = .0185f;//.502f;
		knot[i].tc.u = knot[i].tc.v = .502f;
		sum = knot[i].pos.x + knot[i].pos.y + knot[i].pos.z;
		if (sum > max) max = sum;
		else if (sum < min) min = sum;
	}
	scale = 1.f / (max - min);
	for (i = 0; i < cnt; i++) {
		sum = knot[i].pos.x + knot[i].pos.y + knot[i].pos.z;
		knot[i].color.r = knot[i].color.g = knot[i].color.b = (sum - min) * scale;
		//knot[i].tc.u = knot[i].tc.v = (sum - min) * scale;
	}
	fscanf_s(file, "%d\n", &cnt);
	for (i = 0; i < cnt; i++) {
		fscanf_s(file, "%d %d %d\n", &knot_index[i * 3], &knot_index[i * 3 + 1], &knot_index[i * 3 + 2]);
		p1 = &knot[knot_index[i * 3]].pos; p2 = &knot[knot_index[i * 3 + 1]].pos; p3 = &knot[knot_index[i * 3 + 2]].pos;
		vector_sub(&v1, p2, p1); vector_sub(&v2, p3, p1);
		vector_normalize(vector_crossproduct(&v1, &v1, &v2));
		v1.w = 0.f;
#ifdef NORMAL_INTERP
		if (vector_length(&knot[knot_index[i * 3]].normal) > 1e-6) {
			vector_add(&knot[knot_index[i * 3]].normal, &knot[knot_index[i * 3]].normal, &v1);
			vector_normalize(&knot[knot_index[i * 3]].normal); knot[knot_index[i * 3]].normal.w = 0.f;
		}
		else knot[knot_index[i * 3]].normal = v1;
		if (vector_length(&knot[knot_index[i * 3 + 1]].normal) > 1e-6) {
			vector_add(&knot[knot_index[i * 3 + 1]].normal, &knot[knot_index[i * 3 + 1]].normal, &v1);
			vector_normalize(&knot[knot_index[i * 3 + 1]].normal); knot[knot_index[i * 3 + 1]].normal.w = 0.f;
		}
		else knot[knot_index[i * 3 + 1]].normal = v1;
		if (vector_length(&knot[knot_index[i * 3 + 2]].normal) > 1e-6) {
			vector_add(&knot[knot_index[i * 3 + 2]].normal, &knot[knot_index[i * 3 + 2]].normal, &v1);
			vector_normalize(&knot[knot_index[i * 3 + 2]].normal); knot[knot_index[i * 3 + 2]].normal.w = 0.f;
		}
		else knot[knot_index[i * 3 + 2]].normal = v1;
#else
		knot[knot_index[i * 3]].normal = knot[knot_index[i * 3 + 1]].normal = knot[knot_index[i * 3 + 2]].normal = v1;
#endif
	}
	fclose(file);
}

void light_transform(const transform_t* ts, float v[VL], const float* p) {
	matrix_apply((vector_t*)v, (vector_t*)p, &ts->o2l);
	v[0] = v[0] * .083f * sm_res;
	v[1] = v[1] * .083f * sm_res;
}

void shadowmapPS(device_t* device, color_t* o, scanline_t* sl, IUINT32* fb, float* zb, int x) {
	int idx = sl->y * sm_res + x;
	if (sl->v[2] < shadowmap[idx]) shadowmap[idx] = sl->v[2];
}

void shadowmap_begin(device_t* device) {
	memset(shadowmap, 127, sizeof(float) * sm_len);
	transform_func = light_transform;
	VS_func = wireframeVS;
	PS_func = shadowmapPS;
	cull_func = ccw_cull;
	vl = 8;
	src_size = sizeof(float) * vl;
	RS_CLIPPING = 0;
	RS_SCREEN_MAPPING = 0;
	device->render_state = RENDER_STATE_COLOR;
}

void shadowmap_end(device_t* device, int rs) {
	device_set_render_state(device, rs);
}

char shadowmap_test(device_t* device, const point_t* p) {
	int x, y, idx;
	x = (int)(p->x + .5f), y = (int)(p->y + .5f);
	if (x <= 0 || x >= sm_res || y <= 0 || y >= sm_res) return 0;
	idx = y * sm_res + x;
	return ((p->z) - shadowmap[idx]) > (96.f / sm_res);
}

void draw_plane(device_t *device, int a, int b, int c, int d, const vector_t* normal) {
	vertex_t p1 = mesh[a], p2 = mesh[b], p3 = mesh[c], p4 = mesh[d];
	p1.tc.u = 0, p1.tc.v = 0, p2.tc.u = 0, p2.tc.v = 1;
	p3.tc.u = 1, p3.tc.v = 0, p4.tc.u = 1, p4.tc.v = 1;
#ifndef NORMAL_INTERP
	p1.normal = p2.normal = p3.normal = p4.normal = *normal;
#endif
	device_draw_primitive(device, &p1, &p2, &p3);
	device_draw_primitive(device, &p3, &p2, &p4);
}

void draw_box(device_t *device) {
	vector_t normal = { 0.f, 0.f, -1.f, 0.f };
	draw_plane(device, 0, 1, 2, 3, &normal); // front
	normal.z = 1.f;
	draw_plane(device, 6, 7, 4, 5, &normal); // back
	normal.z = 0.f; normal.x = -1.f;
	draw_plane(device, 4, 5, 0, 1, &normal); // left
	normal.x = 1.f;
	draw_plane(device, 2, 3, 6, 7, &normal); // right
	normal.x = 0.f; normal.y = 1.f;
	draw_plane(device, 1, 5, 3, 7, &normal); // top
	normal.y = -1.f;
	draw_plane(device, 4, 0, 6, 2, &normal); // bottom
}

void draw_grid(device_t *device, int oldRS, const matrix_t *m) {
	int i, j;
	matrix_t o = device->transform.world;
	device->transform.world = *m;
	transform_update(&device->transform);
	for (i = 0; i < 8; i++) {
		for (j = 0; j < 8; j++) {
			device_draw_primitive(device, &grid[i * 9 + j], &grid[(i + 1) * 9 + j], &grid[i * 9 + j + 1]);
			device_draw_primitive(device, &grid[i * 9 + j + 1], &grid[(i + 1) * 9 + j], &grid[(i + 1) * 9 + j + 1]);
		}
	}
	device->transform.world = o;
}

void draw_knot(device_t *device) {
	int i;
	static matrix_t t;
	matrix_t o = device->transform.world;
	if (screen_keys[VK_XBUTTON1]) return;
	matrix_set_identity(&t);
	t.m[0][0] = 10.f; t.m[1][1] = 10.f; t.m[2][2] = 10.f; t.m[3][2] = 10.f;
	matrix_mul(&device->transform.world, &o, &t);
	transform_update(&device->transform);
	for (i = 0; i < 46464; i++)
		device_draw_primitive(device, &knot[knot_index[i * 3]], &knot[knot_index[i * 3 + 1]], &knot[knot_index[i * 3 + 2]]);
	device->transform.world = o;
	transform_update(&device->transform);
}

void write_FPS() {
	static char str[4], debug[512];
	static unsigned long last = 0, current = 0, cnt = 0;
	current = timeGetTime();
	if (current - last >= 1000) {
		sprintf_s(str, 4, "%d", cnt);
		last = current; cnt = 0;
	}
	else cnt++;
#ifdef CALL_CNT 
	sprintf_s(debug, 512, "vl %d\nvadd %d\nvsub %d\nvsca %d\nvdot %d\nvcross %d\n\
						  vinterp %d\nvnor %d\nmadd %d\nmsub %d\nmmul %d\nmsca %d\nmapp %d\nmid %d\nmzero %d\n\
						  mtra %d\nmssca %d\nmrot %d\nmproj %d\nmlook %d\n", vl, vadd, vsub, vsca, vdot, vcross,
		vinterp, vnor, madd, msub, mmul, msca, mapp, mid, mzero, mtra, mssca, mrot, mproj, mlook);
	vl = vadd = vsub = vsca = vdot = vcross = vinterp = vnor = madd = msub =
		mmul = msca = mapp = mid = mzero = mtra = mssca = mrot = mproj = mlook = 0;
#endif
	SelectObject(screen_dc, screen_font);
#ifdef CALL_CNT 
	DrawTextA(screen_dc, debug, (int)strlen(debug), &screen_rect, DT_LEFT);
#endif
	DrawTextA(screen_dc, str, (int)strlen(str), &screen_rect, DT_RIGHT);
	SelectObject(screen_dc, screen_hb);
}

void update_camera(matrix_t *m, point_t *eye, point_t *up, vector_t *xaxis, vector_t *yaxis, vector_t *zaxis) {
	zaxis->x = m->m[0][2]; zaxis->y = m->m[1][2]; zaxis->z = m->m[2][2];
	vector_normalize(zaxis);
	if (m->m[1][1] < 0.f) up->y = -1.f;
	else up->y = 1.f;
	vector_crossproduct(xaxis, up, zaxis);
	vector_normalize(xaxis);
	vector_crossproduct(yaxis, zaxis, xaxis);

	m->m[0][0] = xaxis->x;
	m->m[1][0] = xaxis->y;
	m->m[2][0] = xaxis->z;
	m->m[3][0] = -vector_dotproduct(xaxis, eye);

	m->m[0][1] = yaxis->x;
	m->m[1][1] = yaxis->y;
	m->m[2][1] = yaxis->z;
	m->m[3][1] = -vector_dotproduct(yaxis, eye);

	m->m[0][2] = zaxis->x;
	m->m[1][2] = zaxis->y;
	m->m[2][2] = zaxis->z;
	m->m[3][2] = -vector_dotproduct(zaxis, eye);

	m->m[0][3] = 0.f;
	m->m[1][3] = 0.f;
	m->m[2][3] = 0.f;
	m->m[3][3] = 1.f;
}

void apply_inverse(vector_t* v, const matrix_t* m) {
	vector_t b = *v;
	v->x = b.x * m->m[0][0] + b.y * m->m[0][1] + b.z * m->m[0][2];
	v->y = b.x * m->m[1][0] + b.y * m->m[1][1] + b.z * m->m[1][2];
	v->z = b.x * m->m[2][0] + b.y * m->m[2][1] + b.z * m->m[2][2];
}

int main()
{
	device_t device;
	char indicator = 3, kbhit = 0, cnt = 100;
	float alpha = 0.f;
	point_t eye = { 0.f, 0.f, -4.f }, at = { .1f, .1f, .1f, 1.f }, up = { 0.f, 1.f, 0.f, 1.f };
	vector_t xaxis, yaxis, zaxis, xt, yt, zt;
	matrix_t m1, m2, id;

	TCHAR *title = _T("Software Renderer");

	if (screen_init(1280, 720, title))
		return -1;

	device_init(&device, 1280, 720, screen_fb);

	init_knot();
	init_grid();
	init_texture(&device);
	device_set_render_state(&device, states[indicator]);

	matrix_set_identity(&m1); matrix_set_identity(&m2); matrix_set_identity(&id);
	matrix_set_rotate(&device.transform.world, 1.f, -.5f, 1.f, 1.4f);
	matrix_set_lookat(&device.transform.view, &eye, &at, &up);
	screen_keys[VK_XBUTTON2] = 1;

	ShowWindow(screen_handle, SW_NORMAL);
	while (screen_exit == 0 && screen_keys[VK_ESCAPE] == 0) {
		screen_dispatch();
		device_clear(&device);
		if (screen_keys['W']) vector_add(&eye, &eye, vector_scale(&zt, &zaxis, .1f));
		else if (screen_keys['S']) vector_sub(&eye, &eye, vector_scale(&zt, &zaxis, .1f));
		if (screen_keys['A']) vector_sub(&eye, &eye, vector_scale(&xt, &xaxis, .1f));
		else if (screen_keys['D']) vector_add(&eye, &eye, vector_scale(&xt, &xaxis, .1f));
		if (screen_keys['Q']) vector_sub(&eye, &eye, vector_scale(&yt, &up, .1f));
		else if (screen_keys['E']) vector_add(&eye, &eye, vector_scale(&yt, &up, .1f));
		if (screen_keys[VK_UP]) {
			matrix_set_rotate(&m2, 1.f, 0.f, 0.f, .01f);
			matrix_mul(&device.transform.view, &device.transform.view, &m2);
		}
		else if (screen_keys[VK_DOWN]) {
			matrix_set_rotate(&m2, -1.f, 0.f, 0.f, .01f);
			matrix_mul(&device.transform.view, &device.transform.view, &m2);
		}
		if (screen_keys[VK_LEFT]) {
			matrix_set_rotate(&m2, 0.f, 1.f, 0.f, .01f);
			matrix_mul(&device.transform.view, &device.transform.view, &m2);
		}
		else if (screen_keys[VK_RIGHT]) {
			matrix_set_rotate(&m2, 0.f, -1.f, 0.f, .01f);
			matrix_mul(&device.transform.view, &device.transform.view, &m2);
		}
		if (screen_keys[VK_NUMPAD0])
			matrix_set_lookat(&device.transform.view, &eye, &at, &up);

		if (screen_keys[VK_SPACE]) {
			if (kbhit == 0) {
				kbhit = 1;
				if (indicator++ >= 3) indicator = 0;
				device_set_render_state(&device, states[indicator]);
			}
		}
		else kbhit = 0;
		if (screen_keys[VK_LBUTTON]) { // Model Rotation
			vector_t drag = { 0.f, 0.f, 0.f, 0.f };
			screen_keys[VK_XBUTTON2] = 1;
			if (cnt++ > 10) {
				drag_x = mouse_x;
				drag_y = mouse_y;
				m1 = device.transform.world;
				cnt = 0;
			}
			drag.x = (float)drag_y - mouse_y; // (y, -x, 0)
			drag.y = (float)drag_x - mouse_x;
			apply_inverse(&drag, &device.transform.view);
			matrix_set_rotate(&m2, drag.x, drag.y, drag.z,
				(fabsf(drag.x) + fabsf(drag.y) + fabsf(drag.z)) / 100.f);
			matrix_mul(&device.transform.world, &m1, &m2);
		}
		else if (screen_keys[VK_RBUTTON]) { // Camera Rotation
			float x, y;
			if (cnt++ > 10) {
				drag_x = mouse_x;
				drag_y = mouse_y;
				m1 = device.transform.view;
				cnt = 0;
			}
			x = (float)mouse_x - drag_x;
			y = (float)drag_y - mouse_y;
			matrix_set_rotate(&m2, y, -x, 0.f, (fabsf(x) + fabsf(y)) / 100.f);
			matrix_mul(&device.transform.view, &m1, &m2);
		}
		else if (screen_keys[VK_MBUTTON]) {
			vector_t dir = eye;
			float l = vector_length(&dir);
			vector_scale(&dir, &dir, 1 / l);
			if (screen_keys[VK_MBUTTON] > 0) vector_add(&eye, &eye, &dir);
			else if (l > 1.f) vector_sub(&eye, &eye, &dir);
			screen_keys[VK_MBUTTON] = 0;
		}
		else {
			if (screen_keys[VK_NUMPAD1]) alpha -= .01f;
			if (screen_keys[VK_NUMPAD2]) alpha += .01f;
			if (screen_keys[VK_NUMPAD4]) {
				if (sm_res > 8) {
					sm_res -= 8; sm_len = sm_res * sm_res;
					screen_keys[VK_XBUTTON2] = 1;
				}
			}
			if (screen_keys[VK_NUMPAD5]) {
				if (sm_res < 1024) {
					sm_res += 8; sm_len = sm_res * sm_res;
					screen_keys[VK_XBUTTON2] = 1;
				}
			}
			if (fabsf(alpha) > 1e-6) {
				matrix_set_rotate(&m2, .5f, -1.f, 1.f, alpha);
				matrix_mul(&device.transform.world, &device.transform.world, &m2);
				alpha = 0.f;
				screen_keys[VK_XBUTTON2] = 1;
			}
			cnt = 100;
		}

		update_camera(&device.transform.view, &eye, &up, &xaxis, &yaxis, &zaxis);
		transform_update(&device.transform);
		if (screen_keys[VK_XBUTTON2] && device.render_state == RENDER_STATE_TEXTURE) {
			shadowmap_begin(&device);
			draw_box(&device);
			draw_knot(&device);
			shadowmap_end(&device, states[indicator]);
			screen_keys[VK_XBUTTON2] = 0;
		}

		draw_box(&device);
		draw_knot(&device);
		draw_grid(&device, indicator, &id);
		write_FPS();
		screen_update();
		//Sleep(1);
	}
	device_destroy(&device);
	screen_close();
	return 0;
}
