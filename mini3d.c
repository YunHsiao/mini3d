
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include <windows.h>
#include <tchar.h>
#include <MMSystem.h>
#pragma comment(lib,"Winmm.lib")

typedef unsigned int IUINT32;

typedef struct { float m[4][4]; } matrix_t;
typedef struct { float x, y, z, w; } vector_t;
typedef vector_t point_t;

int CMID(int x, int min, int max) { return (x < min) ? min : ((x > max) ? max : x); }
float saturate(float x) { return (x < 0.f) ? 0.f : ((x > 1.f) ? 1.f : x); }

// 计算插值：t 为 [0, 1] 之间的数值
float interp(float x1, float x2, float t) { return x1 + (x2 - x1) * t; }

// | v |
float vector_length(const vector_t *v) {
	float sq = v->x * v->x + v->y * v->y + v->z * v->z;
	return (float)sqrt(sq);
}

// z = x + y
vector_t* vector_add(vector_t *z, const vector_t *x, const vector_t *y) {
	z->x = x->x + y->x;
	z->y = x->y + y->y;
	z->z = x->z + y->z;
	z->w = 1.f;
	return z;
}

// z = x - y
vector_t* vector_sub(vector_t *z, const vector_t *x, const vector_t *y) {
	z->x = x->x - y->x;
	z->y = x->y - y->y;
	z->z = x->z - y->z;
	z->w = 1.f;
	return z;
}

// z = x * t
vector_t* vector_scale(vector_t *z, const vector_t *x, float t) {
	z->x = x->x * t;
	z->y = x->y * t;
	z->z = x->z * t;
	z->w = 1.f;
	return z;
}

// 矢量点乘
float vector_dotproduct(const vector_t *x, const vector_t *y) {
	return x->x * y->x + x->y * y->y + x->z * y->z;
}

// 矢量叉乘
vector_t* vector_crossproduct(vector_t *z, const vector_t *x, const vector_t *y) {
	float m1, m2, m3;
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
	z->x = interp(x1->x, x2->x, t);
	z->y = interp(x1->y, x2->y, t);
	z->z = interp(x1->z, x2->z, t);
	z->w = 1.f;
	return z;
}

// 矢量归一化
vector_t* vector_normalize(vector_t *v) {
	float length = vector_length(v);
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
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++)
			c->m[i][j] = a->m[i][j] + b->m[i][j];
	}
	return c;
}

// c = a - b
matrix_t* matrix_sub(matrix_t *c, const matrix_t *a, const matrix_t *b) {
	int i, j;
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++)
			c->m[i][j] = a->m[i][j] - b->m[i][j];
	}
	return c;
}

// c = a * b
matrix_t* matrix_mul(matrix_t *c, const matrix_t *a, const matrix_t *b) {
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
	return c;
}

// c = a * f
matrix_t* matrix_scale(matrix_t *c, const matrix_t *a, float f) {
	int i, j;
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++)
			c->m[i][j] = a->m[i][j] * f;
	}
	return c;
}

// y = x * m
vector_t* matrix_apply(vector_t *y, const vector_t *x, const matrix_t *m) {
	float X = x->x, Y = x->y, Z = x->z, W = x->w;
	y->x = X * m->m[0][0] + Y * m->m[1][0] + Z * m->m[2][0] + W * m->m[3][0];
	y->y = X * m->m[0][1] + Y * m->m[1][1] + Z * m->m[2][1] + W * m->m[3][1];
	y->z = X * m->m[0][2] + Y * m->m[1][2] + Z * m->m[2][2] + W * m->m[3][2];
	y->w = X * m->m[0][3] + Y * m->m[1][3] + Z * m->m[2][3] + W * m->m[3][3];
	return y;
}

matrix_t* matrix_set_identity(matrix_t *m) {
	m->m[0][0] = m->m[1][1] = m->m[2][2] = m->m[3][3] = 1.f;
	m->m[0][1] = m->m[0][2] = m->m[0][3] = 0.f;
	m->m[1][0] = m->m[1][2] = m->m[1][3] = 0.f;
	m->m[2][0] = m->m[2][1] = m->m[2][3] = 0.f;
	m->m[3][0] = m->m[3][1] = m->m[3][2] = 0.f;
	return m;
}

matrix_t* matrix_set_zero(matrix_t *m) {
	m->m[0][0] = m->m[0][1] = m->m[0][2] = m->m[0][3] = 0.f;
	m->m[1][0] = m->m[1][1] = m->m[1][2] = m->m[1][3] = 0.f;
	m->m[2][0] = m->m[2][1] = m->m[2][2] = m->m[2][3] = 0.f;
	m->m[3][0] = m->m[3][1] = m->m[3][2] = m->m[3][3] = 0.f;
	return m;
}

// 平移变换
matrix_t* matrix_set_translate(matrix_t *m, float x, float y, float z) {
	matrix_set_identity(m);
	m->m[3][0] = x;
	m->m[3][1] = y;
	m->m[3][2] = z;
	return m;
}

// 缩放变换
matrix_t* matrix_set_scale(matrix_t *m, float x, float y, float z) {
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
	matrix_t transform;     // transform = world * view * projection
	float w, h;             // 屏幕大小
}	transform_t;

// 矩阵更新，计算 transform = world * view * projection
void transform_update(transform_t *ts) {
	matrix_t m;
	matrix_mul(&m, &ts->world, &ts->view);
	matrix_mul(&ts->transform, &m, &ts->projection);
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

// 将矢量 x 进行 project 
void transform_apply(const transform_t *ts, vector_t *y, const vector_t *x) {
	matrix_apply(y, x, &ts->transform);
}

// 检查齐次坐标同 cvv 的边界用于视锥裁剪
int transform_check_cvv(const vector_t *v) {
	float w = v->w;
	int check = 0;
	if (v->z < 0.f) check |= 1;
	if (v->z > w) check |= 2;
	if (v->x < -w) check |= 4;
	if (v->x > w) check |= 8;
	if (v->y < -w) check |= 16;
	if (v->y > w) check |= 32;
	return check;
}

// 归一化，得到屏幕坐标
void transform_homogenize(const transform_t *ts, vector_t *y, const vector_t *x, float rhw) {
	y->x = (x->x * rhw + 1.f) * ts->w * .5f;
	y->y = (1.f - x->y * rhw) * ts->h * .5f;
	y->z = x->z * rhw;
	y->w = 1.f;
}

typedef struct { float r, g, b; } color_t;
typedef struct { float u, v; } texcoord_t;
typedef struct { point_t pos; vector_t normal; texcoord_t tc; color_t color; float rhw; } vertex_t;

typedef struct { vertex_t v, v1, v2; } edge_t;
typedef struct { float top, bottom; edge_t left, right; } trapezoid_t;
typedef struct { vertex_t v, step; int x, y, w; } scanline_t;

//#define NORMAL_INTERP

void vertex_rhw_init(vertex_t *v) {
	float rhw = v->rhw;
#ifdef NORMAL_INTERP
	v->normal.x *= rhw;
	v->normal.y *= rhw;
	v->normal.z *= rhw;
#endif
	v->tc.u *= rhw;
	v->tc.v *= rhw;
	v->color.r *= rhw;
	v->color.g *= rhw;
	v->color.b *= rhw;
}

void vertex_interp(vertex_t *y, const vertex_t *x1, const vertex_t *x2, float t) {
	vector_interp(&y->pos, &x1->pos, &x2->pos, t);
	vector_interp(&y->normal, &x1->normal, &x2->normal, t);
	y->tc.u = interp(x1->tc.u, x2->tc.u, t);
	y->tc.v = interp(x1->tc.v, x2->tc.v, t);
	y->color.r = interp(x1->color.r, x2->color.r, t);
	y->color.g = interp(x1->color.g, x2->color.g, t);
	y->color.b = interp(x1->color.b, x2->color.b, t);
	y->rhw = interp(x1->rhw, x2->rhw, t);
}

void vertex_division(vertex_t *y, const vertex_t *x1, const vertex_t *x2, float w) {
	float inv = 1.f / w;
	y->pos.x = (x2->pos.x - x1->pos.x) * inv;
	y->pos.y = (x2->pos.y - x1->pos.y) * inv;
	y->pos.z = (x2->pos.z - x1->pos.z) * inv;
	y->pos.w = (x2->pos.w - x1->pos.w) * inv;
#ifdef NORMAL_INTERP
	y->normal.x = (x2->normal.x - x1->normal.x) * inv;
	y->normal.y = (x2->normal.y - x1->normal.y) * inv;
	y->normal.z = (x2->normal.z - x1->normal.z) * inv;
	y->normal.w = (x2->normal.w - x1->normal.w) * inv;
#endif
	y->tc.u = (x2->tc.u - x1->tc.u) * inv;
	y->tc.v = (x2->tc.v - x1->tc.v) * inv;
	y->color.r = (x2->color.r - x1->color.r) * inv;
	y->color.g = (x2->color.g - x1->color.g) * inv;
	y->color.b = (x2->color.b - x1->color.b) * inv;
	y->rhw = (x2->rhw - x1->rhw) * inv;
}

void vertex_add(vertex_t *y, const vertex_t *x) {
	y->pos.x += x->pos.x;
	y->pos.y += x->pos.y;
	y->pos.z += x->pos.z;
	y->pos.w += x->pos.w;
#ifdef NORMAL_INTERP
	y->normal.x += x->normal.x;
	y->normal.y += x->normal.y;
	y->normal.z += x->normal.z;
	y->normal.w += x->normal.w;
#endif
	y->rhw += x->rhw;
	y->tc.u += x->tc.u;
	y->tc.v += x->tc.v;
	y->color.r += x->color.r;
	y->color.g += x->color.g;
	y->color.b += x->color.b;
}

// 根据三角形生成 0-2 个梯形，并且返回合法梯形的数量
int trapezoid_init_triangle(trapezoid_t *trap, const vertex_t *p1,
	const vertex_t *p2, const vertex_t *p3) {
	const vertex_t *p;
	float k, x;

	if (p1->pos.y > p2->pos.y) p = p1, p1 = p2, p2 = p;
	if (p1->pos.y > p3->pos.y) p = p1, p1 = p3, p3 = p;
	if (p2->pos.y > p3->pos.y) p = p2, p2 = p3, p3 = p;
	if (p1->pos.y == p2->pos.y && p1->pos.y == p3->pos.y) return 0;
	if (p1->pos.x == p2->pos.x && p1->pos.x == p3->pos.x) return 0;

	if (p1->pos.y == p2->pos.y) { // down
		if (p1->pos.x > p2->pos.x) p = p1, p1 = p2, p2 = p;
		trap[0].top = p1->pos.y;
		trap[0].bottom = p3->pos.y;
		trap[0].left.v1 = *p1;
		trap[0].left.v2 = *p3;
		trap[0].right.v1 = *p2;
		trap[0].right.v2 = *p3;
		return (trap[0].top < trap[0].bottom) ? 1 : 0;
	}

	if (p2->pos.y == p3->pos.y) { // up
		if (p2->pos.x > p3->pos.x) p = p2, p2 = p3, p3 = p;
		trap[0].top = p1->pos.y;
		trap[0].bottom = p3->pos.y;
		trap[0].left.v1 = *p1;
		trap[0].left.v2 = *p2;
		trap[0].right.v1 = *p1;
		trap[0].right.v2 = *p3;
		return (trap[0].top < trap[0].bottom) ? 1 : 0;
	}

	trap[0].top = p1->pos.y;
	trap[0].bottom = p2->pos.y;
	trap[1].top = p2->pos.y;
	trap[1].bottom = p3->pos.y;

	k = (p3->pos.y - p1->pos.y) / (p2->pos.y - p1->pos.y);
	x = p1->pos.x + (p2->pos.x - p1->pos.x) * k;

	if (x <= p3->pos.x) { // left
		trap[0].left.v1 = *p1;
		trap[0].left.v2 = *p2;
		trap[0].right.v1 = *p1;
		trap[0].right.v2 = *p3;
		trap[1].left.v1 = *p2;
		trap[1].left.v2 = *p3;
		trap[1].right.v1 = *p1;
		trap[1].right.v2 = *p3;
	}
	else { // right
		trap[0].left.v1 = *p1;
		trap[0].left.v2 = *p3;
		trap[0].right.v1 = *p1;
		trap[0].right.v2 = *p2;
		trap[1].left.v1 = *p1;
		trap[1].left.v2 = *p3;
		trap[1].right.v1 = *p2;
		trap[1].right.v2 = *p3;
	}

	return 2;
}

// 按照 Y 坐标计算出左右两条边纵坐标等于 Y 的顶点
void trapezoid_edge_interp(trapezoid_t *trap, float y) {
	float s1 = trap->left.v2.pos.y - trap->left.v1.pos.y;
	float s2 = trap->right.v2.pos.y - trap->right.v1.pos.y;
	float t1 = (y - trap->left.v1.pos.y) / s1;
	float t2 = (y - trap->right.v1.pos.y) / s2;
	vertex_interp(&trap->left.v, &trap->left.v1, &trap->left.v2, t1);
	vertex_interp(&trap->right.v, &trap->right.v1, &trap->right.v2, t2);
}

// 根据左右两边的端点，初始化计算出扫描线的起点和步长
void trapezoid_init_scan_line(const trapezoid_t *trap, scanline_t *scanline, int y) {
	float width = trap->right.v.pos.x - trap->left.v.pos.x;
	scanline->x = (int)(trap->left.v.pos.x + .5f);
	scanline->w = (int)(trap->right.v.pos.x + .5f) - scanline->x;
	scanline->y = y;
	scanline->v = trap->left.v;
	if (trap->left.v.pos.x >= trap->right.v.pos.x) scanline->w = 0;
	vertex_division(&scanline->step, &trap->left.v, &trap->right.v, width);
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
	color_t ambient;			// 环境光颜色
}	device_t;

#define RENDER_STATE_POINT	        1		// 渲染点阵
#define RENDER_STATE_WIREFRAME      2		// 渲染线框
#define RENDER_STATE_COLOR	        4		// 渲染颜色
#define RENDER_STATE_TEXTURE        8		// 渲染纹理

// 设备初始化，fb为外部帧缓存，非 NULL 将引用外部帧缓存（每行 4字节对齐）
void device_init(device_t *device, int width, int height, void *fb) {
	int need = sizeof(void*) * (height * 2 + 1024) + width * height * 8;
	char *ptr = (char*)malloc(need + 64);
	char *framebuf, *zbuf;
	int j;
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
	device->ambient.r = .15f;
	device->ambient.g = .15f;
	device->ambient.b = .15f;
	transform_init(&device->transform, width, height);
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
void device_clear(device_t *device, char greyscale) {
	int y, x, height = device->height;
	IUINT32 cc = device->background, *dstf;
	float *dstd;
	for (y = 0; y < device->height; y++) {
		if (greyscale) {
			cc = (height - 1 - y) * 230 / (height - 1);
			cc = (cc << 16) | (cc << 8) | cc;
		}
		dstf = device->framebuffer[y];
		for (x = device->width; x > 0; dstf++, x--) dstf[0] = cc;
	}
	for (y = 0; y < device->height; y++) {
		dstd = device->zbuffer[y];
		for (x = device->width; x > 0; dstd++, x--) dstd[0] = 0.f;
	}
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
void device_draw_line(device_t *device, const vertex_t *v1, const vertex_t *v2, IUINT32 c) {
	int x1 = (int)v1->pos.x, x2 = (int)v2->pos.x, y1 = (int)v1->pos.y, y2 = (int)v2->pos.y;
	float rhw = v1->rhw, rhwstp = (v2->rhw - v1->rhw);
	int x, y, rem = 0, offset;
	if (x1 == x2 && y1 == y2) {
		device_pixel(device, x1, y1, c, v1->rhw);
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
			if (x2 < x1) x = x1, y = y1, x1 = x2, y1 = y2, x2 = x, y2 = y, rhw = v2->rhw, rhwstp = -rhwstp;
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
			if (y2 < y1) x = x1, y = y1, x1 = x2, y1 = y2, x2 = x, y2 = y, rhw = v2->rhw, rhwstp = -rhwstp;
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
	int x, y; float dx, dy, wx, wy;
	IUINT32 c00, c01, c10, c11;
	IUINT32 r00, r01, r10, r11, g00, g01, g10, g11, b00, b01, b10, b11;
	u = u * (device->max_u - 1);
	v = v * (device->max_v - 1);
	x = (int)u;
	y = (int)v;
	x = CMID((int)(u), 0, device->tex_width - 2); // Clamp
	y = CMID((int)(v), 0, device->tex_height - 2);
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
}

// 绘制扫描线
void device_draw_scanline(device_t *device, scanline_t *scanline) {
	IUINT32 *framebuffer = device->framebuffer[scanline->y];
	float *zbuffer = device->zbuffer[scanline->y];
	int x = scanline->x;
	int w = scanline->w;
	int width = device->width;
	int render_state = device->render_state;
	int R, G, B;
	color_t cc;
	float cDiffuse;
	vector_t Nw;
#ifdef NORMAL_INTERP
	float cSpecular;
	vector_t Vw, Hw;
	Vw.x = device->transform.view.m[0][2];
	Vw.y = device->transform.view.m[1][2];
	Vw.z = device->transform.view.m[2][2];
#endif
	for (; w > 0; x++, w--) {
		if (x >= 0 && x < width) {
			float rhw = scanline->v.rhw;
			if (rhw >= zbuffer[x]) {
				float w = 1.f / rhw;
				zbuffer[x] = rhw;
				if (render_state & RENDER_STATE_COLOR) {
					cc.r = scanline->v.color.r * w * 255;
					cc.g = scanline->v.color.g * w * 255;
					cc.b = scanline->v.color.b * w * 255;
				}
				if (render_state & RENDER_STATE_TEXTURE) {
					float u = scanline->v.tc.u *w;
					float v = scanline->v.tc.v *w;
					device_texture_read(&cc, device, u, v);
					// per-pixel Blinn-Phong Illumination
					Nw = scanline->v.normal;
					vector_normalize(&Nw);
					cDiffuse = saturate(-vector_dotproduct(&device->light_dir, &Nw));
					cc.r *= device->ambient.r + cDiffuse;
					cc.g *= device->ambient.g + cDiffuse;
					cc.b *= device->ambient.b + cDiffuse;
#ifdef NORMAL_INTERP
					vector_add(&Hw, &Vw, &device->light_dir);
					vector_normalize(&Hw);
					cSpecular = saturate(powf(vector_dotproduct(&Nw, &Hw), 400)) * 255;
					cc.r += cSpecular; cc.g += cSpecular; cc.b += cSpecular;
#endif
				}

				R = CMID((int)cc.r, 0, 255);
				G = CMID((int)cc.g, 0, 255);
				B = CMID((int)cc.b, 0, 255);
				framebuffer[x] = (R << 16) | (G << 8) | (B);
			}
		}
		vertex_add(&scanline->v, &scanline->step);
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
		else break;
	}
}

// 根据 render_state 绘制原始三角形
void device_draw_primitive(device_t *device, const vertex_t *v1,
	const vertex_t *v2, const vertex_t *v3) {
	vertex_t t1 = *v1, t2 = *v2, t3 = *v3;
	int render_state = device->render_state;
	vector_t a1, a2;

	transform_apply(&device->transform, &t1.pos, &v1->pos);
	transform_apply(&device->transform, &t2.pos, &v2->pos);
	transform_apply(&device->transform, &t3.pos, &v3->pos);
	matrix_apply(&t1.normal, &v1->normal, &device->transform.world);
	matrix_apply(&t2.normal, &v2->normal, &device->transform.world);
	matrix_apply(&t3.normal, &v3->normal, &device->transform.world);

	// 裁剪
	if (transform_check_cvv(&t1.pos)) return;
	if (transform_check_cvv(&t2.pos)) return;
	if (transform_check_cvv(&t3.pos)) return;

	t1.rhw = 1.f / t1.pos.w; t2.rhw = 1.f / t2.pos.w; t3.rhw = 1.f / t3.pos.w;
	transform_homogenize(&device->transform, &t1.pos, &t1.pos, t1.rhw);
	transform_homogenize(&device->transform, &t2.pos, &t2.pos, t2.rhw);
	transform_homogenize(&device->transform, &t3.pos, &t3.pos, t3.rhw);

	// CCW背面剔除
	vector_sub(&a1, &t2.pos, &t1.pos);
	vector_sub(&a2, &t3.pos, &t1.pos);
	if (a1.x*a2.y - a2.x*a1.y < 0) return;

	if (render_state & RENDER_STATE_POINT) { // 点绘制
		device_draw_line(device, &t1, &t1, device->foreground);
		device_draw_line(device, &t2, &t2, device->foreground);
		device_draw_line(device, &t3, &t3, device->foreground);
	}
	else if (render_state & RENDER_STATE_WIREFRAME) { // 线框绘制
		device_draw_line(device, &t1, &t2, device->foreground);
		device_draw_line(device, &t2, &t3, device->foreground);
		device_draw_line(device, &t3, &t1, device->foreground);
	}
	else { // 纹理或者色彩绘制
		trapezoid_t traps[2];
		int n;
		vertex_rhw_init(&t1);
		vertex_rhw_init(&t2);
		vertex_rhw_init(&t3);

		n = trapezoid_init_triangle(traps, &t1, &t2, &t3);
		if (n >= 1) device_render_trap(device, &traps[0]);
		if (n >= 2) device_render_trap(device, &traps[1]);
	}
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

	AdjustWindowRect(&rect, GetWindowLongPtr(screen_handle, GWL_STYLE), 0);
	wx = rect.right - rect.left;
	wy = rect.bottom - rect.top;
	sx = (GetSystemMetrics(SM_CXSCREEN) - wx) / 2;
	sy = (GetSystemMetrics(SM_CYSCREEN) - wy) / 2;
	if (sy < 0) sy = 0;
	SetWindowPos(screen_handle, NULL, sx, sy, wx, wy, (SWP_NOCOPYBITS | SWP_NOZORDER | SWP_SHOWWINDOW));
	SetForegroundWindow(screen_handle);

	ShowWindow(screen_handle, SW_NORMAL);
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
	{ { -1.f, -1.f, -1.f, 1.f },{ -1.f, -1.f, -1.f, 0.f },{ 0.f, 0.f },{ 1.f, .2f, .2f }, 1.f },
	{ { -1.f,  1.f, -1.f, 1.f },{ -1.f,  1.f, -1.f, 0.f },{ 0.f, 1.f },{ .2f, 1.f, .2f }, 1.f },
	{ { 1.f, -1.f, -1.f, 1.f },{ 1.f, -1.f, -1.f, 0.f },{ 1.f, 1.f },{ .2f, .2f, 1.f }, 1.f },
	{ { 1.f,  1.f, -1.f, 1.f },{ 1.f,  1.f, -1.f, 0.f },{ 1.f, 0.f },{ 1.f, .2f, 1.f }, 1.f },
	{ { -1.f, -1.f,  1.f, 1.f },{ -1.f, -1.f,  1.f, 0.f },{ 0.f, 0.f },{ 1.f, 1.f, .2f }, 1.f },
	{ { -1.f,  1.f,  1.f, 1.f },{ -1.f,  1.f,  1.f, 0.f },{ 0.f, 1.f },{ .2f, 1.f, 1.f }, 1.f },
	{ { 1.f, -1.f,  1.f, 1.f },{ 1.f, -1.f,  1.f, 0.f },{ 1.f, 1.f },{ 1.f, .3f, .3f }, 1.f },
	{ { 1.f,  1.f,  1.f, 1.f },{ 1.f,  1.f,  1.f, 0.f },{ 1.f, 0.f },{ .2f, 1.f, .3f }, 1.f },
};
unsigned int knot_index[139392];
char states[] = { RENDER_STATE_POINT, RENDER_STATE_WIREFRAME, RENDER_STATE_COLOR, RENDER_STATE_TEXTURE };

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

void init_grid() {
	int i, j;
	memset(&grid, 0, sizeof(vertex_t) * 81);
	for (i = 0; i < 9; i++) {
		for (j = 0; j < 9; j++) {
			grid[i * 9 + j].pos.x = j * .5f - 2.f;
			grid[i * 9 + j].pos.y = -1.2f;
			grid[i * 9 + j].pos.z = i * .5f - 2.f;
			grid[i * 9 + j].pos.w = 1.f;
			grid[i * 9 + j].normal.y = 1.f;
			grid[i * 9 + j].rhw = 1.f;
		}
	}
}

void init_knot() {
	FILE *file;
	int i, cnt = 0; float max = -2.f, min = 2.f, sum, scale;
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
		knot[i].rhw = 1.f;
		sum = knot[i].pos.x + knot[i].pos.y + knot[i].pos.z;
		if (sum > max) max = sum;
		else if (sum < min) min = sum;
	}
	scale = 1.f / (max - min);
	for (i = 0; i < cnt; i++) {
		sum = knot[i].pos.x + knot[i].pos.y + knot[i].pos.z;
		knot[i].color.r = knot[i].color.g = knot[i].color.b = (sum - min) * scale;
	}
	fscanf_s(file, "%d\n", &cnt);
	for (i = 0; i < cnt; i++) {
		fscanf_s(file, "%d %d %d\n", &knot_index[i * 3], &knot_index[i * 3 + 1], &knot_index[i * 3 + 2]);
	}
	fclose(file);
}

void draw_grid(device_t *device, int oldRS, const matrix_t *m) {
	int i, j;
	matrix_t o = device->transform.world;
	device->transform.world = *m;
	transform_update(&device->transform);
	device->render_state = RENDER_STATE_WIREFRAME;
	for (i = 0; i < 8; i++) {
		for (j = 0; j < 8; j++) {
			device_draw_primitive(device, &grid[i * 9 + j], &grid[(i + 1) * 9 + j], &grid[i * 9 + j + 1]);
			device_draw_primitive(device, &grid[i * 9 + j + 1], &grid[(i + 1) * 9 + j], &grid[(i + 1) * 9 + j + 1]);
		}
	}
	device->render_state = states[oldRS];
	device->transform.world = o;
}

void draw_knot(device_t *device, int oldRS) {
	int i;
	static matrix_t t;
	matrix_t o = device->transform.world;
	if (screen_keys[VK_XBUTTON1]) return;
	matrix_set_identity(&t);
	t.m[0][0] = 3.f; t.m[1][1] = 3.f; t.m[2][2] = 3.f; t.m[3][1] = 4.f;
	matrix_mul(&device->transform.world, &o, &t);
	transform_update(&device->transform);
	if (device->render_state == RENDER_STATE_TEXTURE)
		device->render_state = RENDER_STATE_COLOR;
	for (i = 0; i < 46464; i++)
		device_draw_primitive(device, &knot[knot_index[i * 3]], &knot[knot_index[i * 3 + 1]], &knot[knot_index[i * 3 + 2]]);
	device->render_state = states[oldRS];
	device->transform.world = o;
}

void write_FPS() {
	static char str[4];
	static unsigned long last = 0, current = 0, cnt = 0;
	current = timeGetTime();
	if (current - last >= 1000) {
		sprintf_s(str, 4, "%d", cnt);
		last = current; cnt = 0;
	}
	else cnt++;
	SelectObject(screen_dc, screen_font);
	DrawTextA(screen_dc, str, strlen(str), &screen_rect, DT_RIGHT);
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
	device.render_state = states[indicator];

	matrix_set_identity(&m1); matrix_set_identity(&m2); matrix_set_identity(&id);
	matrix_set_rotate(&device.transform.world, 1.f, -.5f, 1.f, 1.4f);
	matrix_set_lookat(&device.transform.view, &eye, &at, &up);

	while (screen_exit == 0 && screen_keys[VK_ESCAPE] == 0) {
		screen_dispatch();
		device_clear(&device, 0);
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
				device.render_state = states[indicator];
			}
		}
		else kbhit = 0;
		if (screen_keys[VK_LBUTTON]) { // Model Rotation
			vector_t drag = { 0.f, 0.f, 0.f, 0.f };
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
			if (fabsf(alpha) > 1e-6) {
				matrix_set_rotate(&m2, .5f, -1.f, 1.f, alpha);
				matrix_mul(&device.transform.world, &device.transform.world, &m2);
				alpha = 0.f;
			}
			cnt = 100;
		}

		update_camera(&device.transform.view, &eye, &up, &xaxis, &yaxis, &zaxis);
		transform_update(&device.transform);
		draw_box(&device);
		draw_knot(&device, indicator);
		draw_grid(&device, indicator, &id);
		write_FPS();
		screen_update();
		//Sleep(1);
	}
	device_destroy(&device);
	screen_close();
	return 0;
}
