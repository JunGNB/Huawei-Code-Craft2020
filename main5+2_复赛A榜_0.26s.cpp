#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <string>
#include <cstring>
#include <fcntl.h> 
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <ctime>
#include <vector>
#include <map>
#include <unordered_map>
#include <queue>
#include <algorithm>
#include <set>
#include <unordered_set>
#include <bitset>
#include <thread>
#include <pthread.h>
#include <atomic>

using namespace std;

#define TEST


#define THREADNUM1 4	// 数据加载线程数
#define THREADNUM2 4	// 找环线程数
#define THREADNUM3 4    // outTrans线程数
#define NODENUM 2000008	// 节点个数
#define EDGENUM 2000008	// 转账数
#define MT_EDGENUM  EDGENUM / THREADNUM1 * 3	// 每个线程的边的内存大小
#define CHARLEN 12		// id字符的长度(10bit+1bit逗号+1bit字符长度)

#define INTMAX		2147483647
#define INTMAX_3	715827882	// INTMAX/3
#define INTMAX_5	429496729	// INTMAX/5

#define LOOPSIZE3 1024 * 512 * 3 * 12 * 10	// 环路节点为3的环的字符总长度
#define LOOPSIZE4 1024 * 512 * 4 *12 * 10
#define LOOPSIZE5 1024 * 1024 * 5 * 12 * 10
#define LOOPSIZE6 1024 * 1024 * 6 * 12 * 10
#define LOOPSIZE7 1024 * 1024 * 7 * 12 * 10

// #define MUT3(a) (((long long)a<<1)+a)
// #define MUT5(a) (((long long)a<<2)+a)
#define MUT3(a) (a * 3ll)
#define MUT5(a) (a * 5ll)


#ifdef TEST
string inputFiles = "./test_data.N111314.E200W.A19630345.txt";
// string inputFiles = "./CG_K16_3.txt";
// string inputFiles = "./CG_K18_3.txt";
// string inputFiles = "./test_data3512444.txt";
// string inputFiles = "./test_data100w.txt";
//string inputFiles = "./test_data.txt";
string outputFiles = "./myresults3.txt";
#else
#define LINUX_RN		// linux格式 \r\n
string inputFiles = "/data/test_data.txt";
string outputFiles = "/projects/student/result.txt";
#endif

struct Edge {
	uint32_t u;
	uint32_t v;
	uint32_t w;
};
// X->Y: 0.2 <= Y/X <= 3   ==>  3X >= Y; 5Y >= X;
struct GNode {
	uint32_t nodeId;
	uint32_t weight;  //当前边的权值
};

struct Node {
	GNode* stNode; //记录该节点边的其实位置
	GNode* fnNode;
	int lens;
};

int thread_ids[16] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };
// mmap读取参数
struct Edge m_edge[EDGENUM];		// 边的集合
//struct Edge mid_edge[EDGENUM];		// 边的集合
//struct Edge* m_edge = m_edge_t;

struct Edge edge_mt[THREADNUM1][MT_EDGENUM];	// 多线程的边
int mt_edgeSize[THREADNUM1] = { 0 };
int m_edgeSize;						// 边的数量
int readBufSize;					// 读取buf的长度

/*********************************************** 数据加载 ********************************************/
struct loadDataParam {
	char* buf;
	struct Edge* edge;
	int& edgeSize;
	int tId;
};

inline void swap(Edge* a, Edge* b) {
	Edge tmp = *a;
	*a = *b;
	*b = tmp;
}

void quick_sort(Edge* beg, Edge* end) {
	if (beg >= end)
		return;
	Edge* left, * right, * i;
	i = left = beg;
	right = end;
	while (i <= right) {
		if (i->v < left->v) {
			swap(i++, left++);
		}
		else if (i->v > left->v) {
			swap(i, right--);
		}
		else {
			++i;
		}
	}
	quick_sort(beg, left - 1);
	quick_sort(i, end);
}

void sortMerge()        //排序合并函数
{
	Edge* index[THREADNUM1], * index_f[THREADNUM1];

	for (int i = 0; i < THREADNUM1; ++i)
	{
		index[i] = edge_mt[i];
		index_f[i] = edge_mt[i] + mt_edgeSize[i];
		m_edgeSize += mt_edgeSize[i];

	}
	Edge** min_index;
	uint32_t min_num;

	for (int i = 0; i < m_edgeSize; ++i)
	{
		min_num = UINT32_MAX;
		for (int j = 0; j < THREADNUM1; ++j)
		{
			if ((index[j] != index_f[j]) && (index[j]->v < min_num))
			{
				min_index = &index[j];
				min_num = index[j]->v;
			}
		}
		m_edge[i] = *(*min_index)++;
	}
}

// 多线程加载文件，进行字符转换，存储到 edg 的图中， 第1个值存起始节点，第2个值存终止节点，第3个值为权重，依次存这三个值
void* loadData_mt(void* arg)
{
	struct loadDataParam* ldParams = (struct loadDataParam*)arg;
	char* buf = ldParams->buf;
	struct Edge* edge = ldParams->edge;
	int tId = ldParams->tId;

	int stbuf = tId * readBufSize / THREADNUM1; int fnbuf;
	if (tId == (THREADNUM1 - 1)) fnbuf = readBufSize;
	else fnbuf = (tId + 1) * readBufSize / THREADNUM1;
	/********** 找到起始终点位置 **********/
	char* ch = NULL;
	if (stbuf == 0) ch = buf;
	else {	// 找到当前行的起始位置作为起始点
		while (buf[stbuf--] != '\n');
		stbuf += 2;
		ch = &buf[stbuf];	// 下一行起始位置
	}
	while (buf[fnbuf--] != '\n');	// 找到当前行的起始位置作为终止点


	/************* 多线程读取数据 *********/
	char* fnEnd = &buf[fnbuf + 2];
	// char *fnEnd = ch + 1000;
	struct Edge* temEdge = edge;
	uint32_t data0 = 0, data1 = 0, data2 = 0;
	char c;
	while (ch < fnEnd) {
		data0 = 0; data1 = 0; data2 = 0;
		while (*ch != ',') {	// 第一个字符
			data0 = (data0 << 1) + (data0 << 3) + (*ch++ - '0');
		}
		++ch;
		while (*ch != ',') {	// 第二个字符
			data1 = (data1 << 1) + (data1 << 3) + (*ch++ - '0');
		}
		++ch;
#ifdef LINUX_RN
		while (*ch != '\r') {
			data2 = (data2 << 1) + (data2 << 3) + (*ch++ - '0');
		}
		ch += 2;
#else
		while (*ch != '\n') {
			data2 = (data2 << 1) + (data2 << 3) + (*ch++ - '0');
		}
		++ch;
#endif
		if (data2) {
			edge->u = data0;  edge->v = data1; (edge++)->w = data2;
		}
	}
	ldParams->edgeSize = (edge - ldParams->edge);

	quick_sort(ldParams->edge, edge - 1);
}

// 数据载入
void loadData()
{
	int fd = open(inputFiles.c_str(), O_RDONLY);
	readBufSize = lseek(fd, 0, SEEK_END);
	char* ch = (char*)mmap(NULL, readBufSize, PROT_READ, MAP_SHARED, fd, 0);
	close(fd);

	struct Edge* mt_edge[THREADNUM1];

	for (int i = 0; i < THREADNUM1; i++) {
		mt_edge[i] = edge_mt[i];	// 读取数据边的集合
	}

	// 多线程数据载入
	pthread_t td[THREADNUM1];
	vector<struct loadDataParam> ldParams; ldParams.reserve(THREADNUM1);
	for (int i = 0; i < THREADNUM1; ++i) {
		struct loadDataParam ldp = { ch, mt_edge[i], mt_edgeSize[i], i };
		ldParams.emplace_back(ldp);
		pthread_create(&td[i], NULL, &loadData_mt, &ldParams[i]);
	}

	for (int i = 0; i < THREADNUM1; ++i) {
		pthread_join(td[i], NULL);
	}

	//多线程合并
	sortMerge();

}

/*********************************************** 多线程建图 *******************************************/

bool cmp(struct Edge& a, struct Edge& b)
{
	return a.v < b.v;
}

bool cmp1(struct GNode& a, struct GNode& b)
{
	return a.nodeId < b.nodeId;
}
//哈希表实现
#define HASHSIZE 0x4000000
#define HASHMOD 0x3ffffff

struct Item {
	uint32_t key;
	int val;
};

Item m_nodeMp[HASHSIZE];  // 数组索引和节点id映射

//插入关键字进散列表
void h_put(uint32_t key, int val)
{
	int addr = key & HASHMOD;
	while (m_nodeMp[addr].val)
	{
		addr = (addr + 1) & HASHMOD;
	}
	m_nodeMp[addr].key = key;
	m_nodeMp[addr].val = val;
}

//查找指定元素
int h_get(uint32_t key)
{
	int addr = key & HASHMOD;
	while (m_nodeMp[addr].key ^ key)
	{
		addr = (addr + 1) & HASHMOD;
		if (!m_nodeMp[addr].val)
		{
			return 0;
		}
	}
	return m_nodeMp[addr].val;
}

// 建图参数
//unordered_map<uint32_t, uint32_t> m_nodeMp;	// 数组索引和节点id映射
uint32_t m_nodeMp_rev[EDGENUM]; //反向索引

Node m_nextNode[EDGENUM];
Node m_lastNode[EDGENUM];

GNode m_nextNode_sm[THREADNUM1][EDGENUM]; //实际存储位置
GNode m_lastNode_sm[THREADNUM1][EDGENUM];

Node mt_nextNode[THREADNUM1][EDGENUM];
Node mt_lastNode[THREADNUM1][EDGENUM];

GNode m_nextNode_st[THREADNUM1][EDGENUM];
GNode m_lastNode_st[THREADNUM1][EDGENUM];


char m_chrSet[NODENUM][CHARLEN];	// 出现的字符串的集合
int m_nodeNum;						// 节点数量

int in_dt[THREADNUM1][EDGENUM], out_dt[THREADNUM1][EDGENUM]; //存储出入度

void* subBuildGragh(void* args) {
	int pthreadNum = *(int*)args;
	int beg = m_edgeSize * pthreadNum / THREADNUM1, end = m_edgeSize * (pthreadNum + 1) / THREADNUM1; //获得需要构建的边
	Node* m_nextNode_pt, * m_lastNode_pt;
	int* in_d = in_dt[pthreadNum];//malloc(m_nodeNum * sizeof(int));
	int* out_d = out_dt[pthreadNum];//malloc(m_nodeNum * sizeof(int));

	m_nextNode_pt = mt_nextNode[pthreadNum];
	m_lastNode_pt = mt_lastNode[pthreadNum];


	m_nextNode_pt[0].stNode = m_nextNode_st[pthreadNum];
	m_lastNode_pt[0].stNode = m_lastNode_st[pthreadNum];


	//计算出入度
	uint32_t u, v, w, w3, w5;
	for (Edge* tmpEdge = m_edge + beg; tmpEdge != m_edge + end; ++tmpEdge) {
		if (u = h_get(tmpEdge->u)) {
			++out_d[u];
			++in_d[h_get(tmpEdge->v)];
		}
	}

	for (int i = 1; i < m_nodeNum; ++i) {
		m_nextNode_pt[i] = { m_nextNode_pt[i - 1].stNode + m_nextNode_pt[i - 1].lens, m_nextNode_pt[i - 1].stNode + m_nextNode_pt[i - 1].lens, out_d[i] };
		m_lastNode_pt[i] = { m_lastNode_pt[i - 1].stNode + m_lastNode_pt[i - 1].lens, m_lastNode_pt[i - 1].stNode + m_lastNode_pt[i - 1].lens, in_d[i] };
	}

	for (Edge* tmpEdge = m_edge + beg; tmpEdge != m_edge + end; ++tmpEdge) {
		if (u = h_get(tmpEdge->u)) {
			v = h_get(tmpEdge->v);
			w = tmpEdge->w;
			*m_nextNode_pt[u].fnNode++ = { v, w }; //利用nodeId存储当前数组的长度
			*m_lastNode_pt[v].fnNode++ = { u, w };
		}
	}

}

void* subMergeGraph(void* args) {
	int pthreadNum = *(int*)args;
	int beg = m_nodeNum * pthreadNum / THREADNUM1, end = m_nodeNum * (pthreadNum + 1) / THREADNUM1; //获得需要构建的边

	char tmp_char[12];
	char index[12] = { 0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39 };
	int curlen = 0, j;
	uint32_t cut_t;
	if (beg == 0)
		beg = 1;

	GNode* curn = m_nextNode_sm[pthreadNum];
	GNode* curl = m_lastNode_sm[pthreadNum];

	for (int i = beg; i < end; ++i) { //出现0的情况可忽略
		m_nextNode[i].stNode = curn;
		m_lastNode[i].stNode = curl;
		for (int j = 0; j < THREADNUM1; ++j) {
			memcpy(m_nextNode[i].stNode + m_nextNode[i].lens, mt_nextNode[j][i].stNode, mt_nextNode[j][i].lens * sizeof(GNode)); m_nextNode[i].lens += mt_nextNode[j][i].lens;
			memcpy(m_lastNode[i].stNode + m_lastNode[i].lens, mt_lastNode[j][i].stNode, mt_lastNode[j][i].lens * sizeof(GNode)); m_lastNode[i].lens += mt_lastNode[j][i].lens;
		}
		curn += m_nextNode[i].lens;
		curl += m_lastNode[i].lens;
		m_nextNode[i].fnNode = curn;
		m_lastNode[i].fnNode = curl;

		sort(m_lastNode[i].stNode, m_lastNode[i].fnNode, cmp1);

		//缓存结点字符串
		curlen = 0; j = 1; cut_t = m_nodeMp_rev[i];
		do {
			tmp_char[curlen++] = index[cut_t % 10];
			cut_t /= 10;
		} while (cut_t);
		m_chrSet[i][0] = curlen + 1;	// 加上逗号
		while (curlen) {
			m_chrSet[i][j++] = tmp_char[--curlen];
		}
		m_chrSet[i][j++] = 0x2C;		// 逗号
	}
}
union next2_lens {
	uint32_t next2;
	uint32_t lens;
};

/*存储两层结点,增强局部性*/
struct GNode2 {
	// uint32_t lens;		// 仅第一数组存储
	uint32_t next;
	uint32_t nextweight;
	// uint32_t next2;
	next2_lens next2_or_lens;
	uint32_t next2weight;
};

struct Node2 {
	GNode2* stNode;
	GNode2* fnNode;
};

// GNode2 node2storage[THREADNUM1][30000000];  //1900w数据集需要得大小为26360082
GNode2 *node2storage[THREADNUM1];
Node2 m_next2Node[2000005];
atomic_int storage_cur_pos(0); //用于存储的负载均衡

void* subStorage(void* args) {
	int pt = *(int*)args;
	GNode* tmpNode, * tmpNode2;
	node2storage[pt] = (GNode2 *)malloc(sizeof(GNode)*30000000);
	GNode2* cur = node2storage[pt], * last;

	int i;
	while (1) {
		i = ++storage_cur_pos;
		if (i >= m_nodeNum) break;
		m_next2Node[i].stNode = cur;
		for (tmpNode = m_nextNode[i].stNode; tmpNode != m_nextNode[i].fnNode; ++tmpNode) {
			last = cur + 1;
			for (tmpNode2 = m_nextNode[tmpNode->nodeId].stNode; tmpNode2 != m_nextNode[tmpNode->nodeId].fnNode; ++tmpNode2) {
				if (tmpNode2->nodeId == i) continue;
				if (tmpNode->weight > MUT5(tmpNode2->weight) || tmpNode2->weight > MUT3(tmpNode->weight)) continue;
				last->next = tmpNode->nodeId; 					last->nextweight = tmpNode->weight;
				last->next2_or_lens.next2 = tmpNode2->nodeId; 	last->next2weight = tmpNode2->weight;
				++last;
			}
			if (last != cur + 1) {
				cur->next2_or_lens.lens = last - cur;	// 第一个结构体存储第一条边，及该边的邻接边的长度
				cur->next = tmpNode->nodeId; cur->nextweight = tmpNode->weight;
				cur = last;
			}
		}
		m_next2Node[i].fnNode = cur;
	}
}

void buildGragh()
{
	// 映射
	int ncnt = 0;
	struct Edge* tmpEdge = m_edge, * endEdge = m_edge + m_edgeSize;
	endEdge->v = (endEdge - 1)->v + 1;
	for (; tmpEdge != endEdge; ++tmpEdge) {
		if ((tmpEdge + 1)->v != tmpEdge->v) {
			h_put(tmpEdge->v, ++ncnt);
			m_nodeMp_rev[ncnt] = tmpEdge->v;
		}
	}
	m_nodeNum = ncnt + 1;

	//多线程建图
	pthread_t td[THREADNUM1];
	for (int i = 0; i < THREADNUM1; ++i) {
		pthread_create(&td[i], NULL, &subBuildGragh, &thread_ids[i]);
	}
	for (int i = 0; i < THREADNUM1; ++i) {
		pthread_join(td[i], NULL);
	}

	//多线程合并
	for (int i = 0; i < THREADNUM1; ++i) {
		pthread_create(&td[i], NULL, &subMergeGraph, &thread_ids[i]);
	}

	for (int i = 0; i < THREADNUM1; ++i) {
		pthread_join(td[i], NULL);
	}

	//存两层结点 多线程
	for (int i = 0; i < THREADNUM1; ++i) {
		pthread_create(&td[i], NULL, &subStorage, &thread_ids[i]);
	}

	for (int i = 0; i < THREADNUM1; ++i) {
		pthread_join(td[i], NULL);
	}

}


/*********************************************** 找环 *******************************************/
int nodeIntSize[5][NODENUM];            // 存储每个节点的环的长度

uint32_t* loopSetC3[THREADNUM2];
uint32_t* loopSetC4[THREADNUM2];
uint32_t* loopSetC5[THREADNUM2];
uint32_t* loopSetC6[THREADNUM2];
uint32_t* loopSetC7[THREADNUM2];

char *outbuffer3, *outbuffer4, *outbuffer5, *outbuffer6;
char *outbuffer73, *outbuffer74, *outbuffer75, *outbuffer76;

uint8_t owners[NODENUM];// 当前节点属的线程

atomic_int cur_pos(0); //用于负载均衡（映射节点从1开始）

void applySpace()
{
	for (int i = 0; i < THREADNUM2; i++) {
		loopSetC3[i] = (uint32_t*)malloc(LOOPSIZE3);
		loopSetC4[i] = (uint32_t*)malloc(LOOPSIZE4);
		loopSetC5[i] = (uint32_t*)malloc(LOOPSIZE5);
		loopSetC6[i] = (uint32_t*)malloc(LOOPSIZE6);
		loopSetC7[i] = (uint32_t*)malloc(LOOPSIZE7);
	}

    outbuffer3 = (char *)malloc(LOOPSIZE3);
    outbuffer4 = (char *)malloc(LOOPSIZE4);
    outbuffer5 = (char *)malloc(LOOPSIZE5);
    outbuffer6 = (char *)malloc(LOOPSIZE6);

    outbuffer73 = (char *)malloc(LOOPSIZE7);
    outbuffer74 = (char *)malloc(LOOPSIZE7);
    outbuffer75 = (char *)malloc(LOOPSIZE7);
    outbuffer76 = (char *)malloc(LOOPSIZE7);
}

inline GNode* upperBound(GNode* A, int end, const int& tar)
{
	int mid; int start = 0;
	while (start < end) {
		mid = (start + end) >> 1;
		(A[mid].nodeId > tar) ? end = mid : start = mid + 1;
	}
	return A + start;
}

inline GNode* lowerBound(GNode* A, int end, const int& tar)
{
	int mid; int start = 0;
	while (start < end) {
		mid = (start + end) >> 1;
		(A[mid].nodeId < tar) ? start = mid + 1 : end = mid;
	}
	return A + start;
}

inline GNode2* lowerBound2(GNode2* A, int end, const int& tar)
{
	int mid; int start = 0;
	while (start < end) {
		mid = (start + end) >> 1;
		(A[mid].next2_or_lens.next2 < tar) ? start = mid + 1 : end = mid;
	}
	return A + start;
}



struct P2List {
	uint32_t* stList;
	uint32_t* fnList;
};

struct Path {
	uint32_t lv6n;
	uint32_t lv6w;
	uint32_t lv7n;
};

struct P3List {
	Path* stList;
	Path* fnList;
};

struct dfsParams {
	int stId;
	uint32_t** loopSet;
	int* adjNodeSet;
	uint32_t* adjWeight;
	int* n2stIndex;
	uint8_t* level;
	struct P3List* p3List;
	bool* instack;
};
void DFS(struct dfsParams* dp)
{
	int& stId = dp->stId; uint32_t** loopSet = dp->loopSet; int* adjNodeSet = dp->adjNodeSet; uint32_t* adjWeight = dp->adjWeight; 
	int* n2stIndex = dp->n2stIndex; uint8_t* level = dp->level; bool* instack = dp->instack; struct P3List* p3List = dp->p3List;

	register int loopV0, loopV1, loopV2, loopV3, loopV4, loopV5, loopV6;
	uint32_t* path; int adjIndex;
	struct GNode *lv6Node, *revNode, *revNodeIndex; struct GNode2 *lv3to5, *lv1to3; 
	struct Path *path3;
	
	loopV0 = stId; instack[loopV0] = 1;	// 第一层 
	revNodeIndex = upperBound(m_lastNode[loopV0].stNode, m_lastNode[loopV0].lens, loopV0);
	
	// if(m_next2Node[loopV0].stNode == m_next2Node[loopV0].fnNode || (m_next2Node[loopV0].fnNode - 1)->next < loopV0) return;
	// lv1to3 = m_next2Node[loopV0].stNode; 
	// while(lv1to3->next <= loopV0) lv1to3 += lv1to3->next2_or_lens.lens;
	
	while ((m_next2Node[loopV0].stNode + n2stIndex[loopV0])->next <= loopV0) 
		n2stIndex[loopV0] += (m_next2Node[loopV0].stNode + n2stIndex[loopV0])->next2_or_lens.lens; // 维护左边界
	lv1to3 = m_next2Node[loopV0].stNode + n2stIndex[loopV0];
	for(; lv1to3 < m_next2Node[loopV0].fnNode;) {
		GNode2 *lv1to3fn = lv1to3 + lv1to3->next2_or_lens.lens;
		if((lv1to3fn - 1)->next2_or_lens.next2 <= loopV0) {	// 若无大于起点的第三层
			lv1to3 += lv1to3->next2_or_lens.lens; // 跳转到下一个节点存储长度位置
			continue;
		}
		// 寻找逆向的满足条件的节点
		adjNodeSet[0] = 0;
		revNode = revNodeIndex;
		for (; revNode != m_lastNode[loopV0].fnNode; ++revNode) {
			if (revNode->weight > MUT5(lv1to3->nextweight) || MUT3(revNode->weight)< lv1to3->nextweight) continue; // 权值判断: X > 5Y 或 3X < Y
			level[revNode->nodeId] |= 0x0f;
			adjNodeSet[++adjNodeSet[0]] = revNode->nodeId;	// 存储满足条件的邻接点
		}
		if (adjNodeSet[0] == 0) {
			lv1to3 += lv1to3->next2_or_lens.lens; // 跳转到下一个节点存储长度位置
			continue; // 无满足条件的边
		}
		
		loopV1 = lv1to3->next; instack[loopV1] = 1;	 
		// ++lv1to3;// 下一个为正式邻接点
		// while(lv1to3->next2_or_lens.next2 <= loopV0) ++lv1to3;
		lv1to3 = lowerBound2(lv1to3 + 1, lv1to3->next2_or_lens.lens - 1, loopV0);
	
		for(; lv1to3 < lv1to3fn; ++lv1to3) { 		
			loopV2 = lv1to3->next2_or_lens.next2;
			if(m_next2Node[loopV2].stNode == m_next2Node[loopV2].fnNode || (m_next2Node[loopV2].fnNode - 1)->next < loopV0) continue;
			instack[loopV2] = 1;  // 第三层
	
			while ((m_next2Node[loopV2].stNode + n2stIndex[loopV2])->next < loopV0) 
				n2stIndex[loopV2] += (m_next2Node[loopV2].stNode + n2stIndex[loopV2])->next2_or_lens.lens; // 维护左边界
			lv3to5 = m_next2Node[loopV2].stNode + n2stIndex[loopV2];
			if(lv3to5->next == loopV0) {
				if (level[loopV2] & 0x08){		// 环数为3的环
					if (lv1to3->next2weight <= MUT5(lv3to5->nextweight) && MUT3(lv1to3->next2weight) >= lv3to5->nextweight) {
						*loopSet[0]++ = loopV0; *loopSet[0]++ = loopV1; *loopSet[0]++ = loopV2;
					}
				}
				lv3to5 += lv3to5->next2_or_lens.lens; // 跳转到下一个节点存储长度位置
			}		
			for(; lv3to5 < m_next2Node[loopV2].fnNode;) {
				if(instack[lv3to5->next] || lv1to3->next2weight > MUT5(lv3to5->nextweight) || MUT3(lv1to3->next2weight) < lv3to5->nextweight) { // 表示该节点不符合要求
					lv3to5 += lv3to5->next2_or_lens.lens; // 跳转到下一个节点存储长度位置
					continue;
				}
				else {
					GNode2 *lv3to5fn = lv3to5 + lv3to5->next2_or_lens.lens;
					if((lv3to5fn - 1)->next2_or_lens.next2 < loopV0) {	// 若无大于起点的第五层
						lv3to5 += lv3to5->next2_or_lens.lens; // 跳转到下一个节点存储长度位置
						continue;
					}
					loopV3 = lv3to5->next; instack[loopV3] = 1; 	// 第四层
					
					if (level[loopV3] & 0x40) { // p3List 有路径
						loopV4 = p3List[loopV3].stList->lv6n - 1;  bool lv4valid = 0;
						for (path3 = p3List[loopV3].stList; path3 != p3List[loopV3].fnList; ++path3) {
							if(path3->lv6n != loopV4) {	// 未访问过
								lv4valid = 0;
								if (lv3to5->nextweight > MUT5(path3->lv6w) || MUT3(lv3to5->nextweight) < path3->lv6w) continue; // 权值判断: X > 5Y 或 3X < Y
								if (instack[path3->lv6n]) continue;
								loopV4 = path3->lv6n; lv4valid = 1;
							}
							if(!lv4valid) continue;
							if ((level[path3->lv7n] & 0x08) && !instack[path3->lv7n]) { // 节点6的环
								loopV5 = path3->lv7n;
								*loopSet[3]++ = loopV0; *loopSet[3]++ = loopV1; *loopSet[3]++ = loopV2; 
								*loopSet[3]++ = loopV3; *loopSet[3]++ = loopV4; *loopSet[3]++ = loopV5;
							}
						}
					}
					
					// ++lv3to5;// 下一个为正式邻接点
					// while(lv3to5->next2_or_lens.next2 < loopV0) ++lv3to5;
					lv3to5 = lowerBound2(lv3to5 + 1, lv3to5->next2_or_lens.lens - 1, loopV0);
					if(lv3to5->next2_or_lens.next2 == loopV0) {
						if ((level[loopV3] & 0x08)){		// 环数为4的环
							*loopSet[1]++ = loopV0; *loopSet[1]++ = loopV1; 
							*loopSet[1]++ = loopV2; *loopSet[1]++ = loopV3;
						}
					}
					
					for(; lv3to5 < lv3to5fn; ++lv3to5) { 			
						if(level[lv3to5->next2_or_lens.next2] && !instack[lv3to5->next2_or_lens.next2]) {
							loopV4 = lv3to5->next2_or_lens.next2; instack[loopV4] = 1;	// 第五层
							if (level[loopV4] & 0x08) {	// 节点数5的环
								if (lv3to5->next2weight <= MUT5(adjWeight[loopV4]) && MUT3(lv3to5->next2weight) >= adjWeight[loopV4]) { // 权值判断: X <= 5Y 或 3X >= Y
									*loopSet[2]++ = loopV0; *loopSet[2]++ = loopV1; 
									*loopSet[2]++ = loopV2; *loopSet[2]++ = loopV3; *loopSet[2]++ = loopV4;
								}
							}
							
							if (level[loopV4] & 0x40) { // p3List 有路径
								loopV5 = p3List[loopV4].stList->lv6n - 1; bool lv5valid = 0;
								for (path3 = p3List[loopV4].stList; path3 != p3List[loopV4].fnList; ++path3) {
									if(path3->lv6n != loopV5) {	// 未访问过
										lv5valid = 0;
										if (lv3to5->next2weight > MUT5(path3->lv6w) || MUT3(lv3to5->next2weight) < path3->lv6w) continue;
										if (instack[path3->lv6n]) continue;
										loopV5 = path3->lv6n; lv5valid = 1;
									}
									if(!lv5valid) continue;
									 // 权值判断: X > 5Y 或 3X < Y
									if ((level[path3->lv7n] & 0x08) && !instack[path3->lv7n]) { // 节点7的环
										loopV6 = path3->lv7n;
										*loopSet[4]++ = loopV0; *loopSet[4]++ = loopV1; *loopSet[4]++ = loopV2; 
										*loopSet[4]++ = loopV3; *loopSet[4]++ = loopV4; *loopSet[4]++ = loopV5; *loopSet[4]++ = loopV6;
									}
								}
							}
							
							instack[loopV4] = 0;
						}
					}
					instack[loopV3] = 0;						
				}
			}
			instack[loopV2] = 0;
		}
		adjIndex = adjNodeSet[0];
		while (adjIndex) {
			level[adjNodeSet[adjIndex--]] &= 0xf7;	// 状态恢复
		}
		
		instack[loopV1] = 0;
	}
}

bool comp3(Path& a, Path& b) {
	if (a.lv6n < b.lv6n)
		return 1;
	else if (a.lv6n == b.lv6n)
		return a.lv7n < b.lv7n;
	else
		return 0;
}

void* findLoop_mt(void* arg)
{
	// #ifdef TEST
	// 	struct timeval tv0, tv1;
	// 	gettimeofday(&tv0, NULL);
	// #endif
	int tId = *(int*)arg;
	
	int* adjNodeSet = (int*)malloc(m_nodeNum * sizeof(int));	// 满足权值要求的逆向邻接点集合
	uint32_t* adjWeight = (uint32_t*)malloc(m_nodeNum * sizeof(uint32_t));	// 逆向邻接权重
	uint32_t* levelNode = (uint32_t*)malloc(m_nodeNum * 2 * sizeof(uint32_t));	// 改变的level节点
	uint32_t* level5Node = (uint32_t*)malloc(m_nodeNum * sizeof(uint32_t));	// 改变的level节点

	int* n2stIndex =  (int*)calloc(m_nodeNum, sizeof(int));		// next2Node起始点索引
	int* lstIndex = (int*)calloc(m_nodeNum, sizeof(int));		// lastNode起始点索引

	struct P3List* p3List = (struct P3List*)malloc(m_nodeNum * sizeof(P3List)); 
	Path* path3 = (Path*)malloc(2000000 * sizeof(Path));	// 存储路径
	
	uint8_t* level = (uint8_t*)malloc(m_nodeNum * sizeof(uint8_t));	// 是否在bfs逆序图中的标志，即是否可能出现在5，6，7层标志
	bool* instack = (bool*)calloc(m_nodeNum, sizeof(bool));		// 是否在栈中的标志

	uint32_t* loopSet[5] = { loopSetC3[tId], loopSetC4[tId], loopSetC5[tId],loopSetC6[tId],loopSetC7[tId] };
	uint32_t* loop3Start, * loop4Start, * loop5Start, * loop6Start, * loop7Start;
	
    struct GNode* lv7Node, * lv6Node, * lv5Node, *lv2Node, *lv2NodeIndex;
	register uint32_t lv7Id, lv6Id, lv5Id;

	int finish = m_nodeNum;  int loopNum = -1; int node, path3Index;
	bool flag, hasloop;// 是否逆向有环	

	struct dfsParams dp = { 0,loopSet, adjNodeSet, adjWeight, n2stIndex, level, p3List, instack };
	
    // 直接找环
	while (true) {
		node = ++cur_pos;
		if (node >= finish) break;

		if (m_lastNode[node].lens && (m_lastNode[node].fnNode - 1)->nodeId > node && 
			m_next2Node[node].stNode != m_next2Node[node].fnNode && (m_next2Node[node].fnNode - 1)->next > node) {
			levelNode[0] = 0; level5Node[0] = 0;flag = 0; path3Index = 0;

			/***********************************************bfs预处理*****************************************************/
			lv2NodeIndex = lowerBound(m_nextNode[node].stNode, m_nextNode[node].lens, node);
			
			while ((m_lastNode[node].stNode + lstIndex[node])->nodeId <= node) ++lstIndex[node];
			lv7Node = m_lastNode[node].stNode + lstIndex[node];
			// lv7Node = upperBound(m_lastNode[node].stNode, m_lastNode[node].lens, node);
			for (; lv7Node != m_lastNode[node].fnNode; ++lv7Node) {
				hasloop = 0;
				for(lv2Node = lv2NodeIndex; lv2Node != m_nextNode[node].fnNode; ++lv2Node) {
					if (lv7Node->weight <= MUT5(lv2Node->weight) && MUT3(lv7Node->weight) >= lv2Node->weight) {
						hasloop = 1;
						break;	// 正逆向的权值有路径
					}
				}
				if(!hasloop) continue;
				
				lv7Id = lv7Node->nodeId; adjWeight[lv7Id] = lv7Node->weight;
				if (!level[lv7Id]) levelNode[++levelNode[0]] = lv7Id;
				// level[lv7Id] |= 0x0f; 

				if (!m_lastNode[lv7Id].lens || (m_lastNode[lv7Id].fnNode - 1)->nodeId <= node) continue;
				while ((m_lastNode[lv7Id].stNode + lstIndex[lv7Id])->nodeId <= node) ++lstIndex[lv7Id];
				lv6Node = m_lastNode[lv7Id].stNode + lstIndex[lv7Id];
				// lv6Node = upperBound(m_lastNode[lv7Node->nodeId].stNode, m_lastNode[lv7Node->nodeId].lens, node);
				for (; lv6Node != m_lastNode[lv7Id].fnNode; ++lv6Node) {
					if (lv6Node->weight > MUT5(lv7Node->weight) || MUT3(lv6Node->weight) < lv7Node->weight) continue; // 权值判断
					lv6Id = lv6Node->nodeId;
					if (!level[lv6Id]) levelNode[++levelNode[0]] = lv6Id;
					level[lv6Id] |= 0x87;
					flag = 1;
// /*
					if (!m_lastNode[lv6Id].lens || (m_lastNode[lv6Id].fnNode - 1)->nodeId <= node) continue;
					while ((m_lastNode[lv6Id].stNode + lstIndex[lv6Id])->nodeId <= node) ++lstIndex[lv6Id];
					lv5Node = m_lastNode[lv6Id].stNode + lstIndex[lv6Id];
					// lv5Node = upperBound(m_lastNode[lv6Node->nodeId].stNode, m_lastNode[lv6Node->nodeId].lens, node);
					for (; lv5Node != m_lastNode[lv6Id].fnNode; ++lv5Node) {
						if (lv5Node->weight > MUT5(lv6Node->weight) || MUT3(lv5Node->weight) < lv6Node->weight) continue; // 权值判断
						lv5Id = lv5Node->nodeId;
						if (!(level[lv5Id] & 0x40)) {
							level[lv5Id] |= 0x43;
							levelNode[++levelNode[0]] = lv5Id;
							level5Node[++level5Node[0]] = lv5Id;	

							p3List[lv5Id].stList = &path3[path3Index];
							p3List[lv5Id].fnList = &path3[path3Index];
							path3Index += 100;

							p3List[lv5Id].fnList->lv6n = lv6Id;
							p3List[lv5Id].fnList->lv6w = lv5Node->weight;
							p3List[lv5Id].fnList++->lv7n = lv7Id;
						}
						else {
							p3List[lv5Id].fnList->lv6n = lv6Id;
							p3List[lv5Id].fnList->lv6w = lv5Node->weight;
							p3List[lv5Id].fnList++->lv7n = lv7Id;
						}
					}
					// */
				}
			}
			/**************************************************************************************************************/
			// dfs开始遍历(逆向有第三层的路径)
			if (flag) {
				for(int i = 1; i <= level5Node[0]; ++i) {
					sort(p3List[level5Node[i]].stList, p3List[level5Node[i]].fnList, comp3);
				}
					
				loop3Start = loopSet[0]; loop4Start = loopSet[1];  loop5Start = loopSet[2];
				loop6Start = loopSet[3]; loop7Start = loopSet[4];
				dp.stId = node; DFS(&dp);

				nodeIntSize[0][node] = (loopSet[0] - loop3Start);				// 存储该节点的环的总长度
				nodeIntSize[1][node] = (loopSet[1] - loop4Start);
				nodeIntSize[2][node] = (loopSet[2] - loop5Start);
				nodeIntSize[3][node] = (loopSet[3] - loop6Start);
				nodeIntSize[4][node] = (loopSet[4] - loop7Start);
			}
			while (levelNode[0]) {
				level[levelNode[levelNode[0]--]] = 0;
			}
		}
		owners[node] = tId;	// 该节点所属线程为tId
	}
	// loopNum_mt[tId] = loopNum + 1;

	// #ifdef TEST
	// 	gettimeofday(&tv1, NULL);
	// 	int t = (tv1.tv_sec - tv0.tv_sec) * 1000 + (tv1.tv_usec - tv0.tv_usec) / 1000.0;
	// 	cout << "@"<<tId<<" clock:"<< t << "ms"<<endl;
	// #endif
}

/*********************************************** 输出 *******************************************/
inline void numToStr(char* str, int num, int& size)
{
	char s[10]; int i = 0, j = 0;

	s[i++] = num % 10 + 0x30;
	num /= 10;
	while (num > 0) {
		s[i++] = num % 10 + 0x30;
		num /= 10;
	}
	size = i;
	while (i) {
		str[j++] = s[--i];
	}
}


void calLoop(int nodeNum[][THREADNUM2], int& loopSum)
{
    // 计算各个找环线程得到的环的节点数量
    int stNodeEnd = m_nodeNum;
    for(int i = 0; i < 5; i++){
        for(int stNodeId = 1; stNodeId < stNodeEnd; stNodeId++){
            nodeNum[i][owners[stNodeId]] += nodeIntSize[i][stNodeId];
        }
    }
    // 计算总环数
    for(int i = 0; i < 5; i++){
        int loopLen = i + 3;
        for(int k = 0; k < THREADNUM2; k++){
            loopSum += nodeNum[i][k] / loopLen;
        }
    }
}


void outTrans_mt(char* outbufferI, uint32_t* loopSet[][THREADNUM2], uint32_t* loopSet7[THREADNUM2], \
     char* outbuffer7, int startId7, int finishId7, int* charSize, int* charSize7, int tId)
{
#ifdef TEST
	struct timeval tv0, tv1;
	gettimeofday(&tv0, NULL);
#endif

    uint32_t** loopSetI = loopSet[tId]; // tId 为outTrans线程号

    char* p0 = outbufferI;
	register int stNodeId = 1; 
    register int fnNodeId = m_nodeNum;
	register int size, owner;

    register int lines = 0;
    register int loopLen = tId + 3;
	for (; stNodeId < fnNodeId; ++stNodeId) {
		size = nodeIntSize[tId][stNodeId];  // 当前节点id环的节点总数目
        if(size){
            lines = size / loopLen;
            owner = owners[stNodeId];
            for(int line = 0; line < lines; line++){    // 将每个节点对应的字符存到buffer中
                for(int n = 0; n < loopLen; n++){
                    // int index = line * loopLen + n;
                    uint32_t id = loopSetI[owner][line * loopLen + n];
                    memcpy(outbufferI, &m_chrSet[id][1], m_chrSet[id][0]);
                    outbufferI += (int)m_chrSet[id][0];
                }
                *(outbufferI - 1) = 0x0A;       // 换行符
            }
            loopSetI[owner] += size;
        }
	}
    charSize[tId] = outbufferI - p0;

	// // 处理7环部分
    p0 = outbuffer7;
	stNodeId = startId7; fnNodeId = finishId7;
    loopLen = 7;
    for (; stNodeId < fnNodeId; ++stNodeId) {
        size = nodeIntSize[4][stNodeId];  // 当前节点id环的节点总数目
        if(size){
            lines = size / loopLen;
            owner = owners[stNodeId];

            for(int line = 0; line < lines; line++){
                for(int n = 0; n < loopLen; n++){
                    // int index = line * loopLen + n;
                    uint32_t id = loopSet7[owner][line * loopLen + n];
                    memcpy(outbuffer7, &m_chrSet[id][1], m_chrSet[id][0]);
                    outbuffer7 += (int)m_chrSet[id][0];
                }
                *(outbuffer7 - 1) = 0x0A;       // 换行符
            }
            loopSet7[owner] += size;
        }
    }
    charSize7[tId] = outbuffer7 - p0;

#ifdef TEST
	gettimeofday(&tv1, NULL);
	cout << "outTrans_mt " << tId << " : " << (tv1.tv_sec - tv0.tv_sec) * 1000 + (tv1.tv_usec - tv0.tv_usec) / 1000.0 << "ms\n";
#endif
}

// void outCharTrans(int loopSum, int& bufSize)
void outCharTrans()
{
#ifdef TEST
    struct timeval tv0, tv1;
    gettimeofday(&tv0, NULL);
#endif

    int loopSum = 0;
    int bufSize = 0;

    int nodeNumSet[5][THREADNUM2] = {0};
    calLoop(nodeNumSet, loopSum);
#ifdef TEST
    cout << "符合规则的环有：" << loopSum << endl;
#endif
    // 第一行为环路个数
    int intSize = 0;
    char *outbuffer3_0 = outbuffer3;
    numToStr(outbuffer3, loopSum, intSize); outbuffer3 += intSize;
    *outbuffer3++ = 0x0A;

	uint32_t* loopSet[5][THREADNUM2];       //  3~7环都保存在loopSet二维数组中
	for (int t = 0; t < THREADNUM2; t++) {
		loopSet[0][t] = loopSetC3[t]; loopSet[1][t] = loopSetC4[t]; loopSet[2][t] = loopSetC5[t];
		loopSet[3][t] = loopSetC6[t]; loopSet[4][t] = loopSetC7[t];
	}

	// 拆分7环到各线程中
    // 计算7环节点数
    int nodeNum3, nodeNum4, nodeNum5, nodeNum6, nodeNum7;
    nodeNum3 = nodeNum4 = nodeNum5 = nodeNum6 = nodeNum7 = 0;
    for(int k = 0; k < THREADNUM2; k++){
        nodeNum3 += nodeNumSet[0][k];
        nodeNum4 += nodeNumSet[1][k];
        nodeNum5 += nodeNumSet[2][k];
        nodeNum6 += nodeNumSet[3][k];
        nodeNum7 += nodeNumSet[4][k];
    }
    // cout << nodeNum3 << " " << nodeNum4 << " " << nodeNum5 << " " << nodeNum6 << " " << nodeNum7 << endl;

    int nodeNum = m_nodeNum;            // 节点ID总数
    int blockPart = (nodeNum3 + nodeNum4 + nodeNum5 + nodeNum6 + nodeNum7) >> 2; // 节点总数分块
    int startId7[4] = {0};
    int finishId7[4] = {0};
    uint32_t* loopSet7_0[THREADNUM2], * loopSet7_1[THREADNUM2], * loopSet7_2[THREADNUM2], * loopSet7_3[THREADNUM2];	// 4线程在7环的起始位置

    int stNodeId = 1; int size;
    // 线程3应处理的长度
    for(int i = 0; i < THREADNUM2; i++)
        loopSet7_3[i] = loopSet[4][i];
    int tdLen = nodeNum6; startId7[3] = 1;
    for (; stNodeId < nodeNum; ++stNodeId) {
        size = nodeIntSize[4][stNodeId];
        if(size){
            loopSet[4][owners[stNodeId]] += size;
            tdLen += size;
            if (tdLen > blockPart) break;
        }
    }
    finishId7[3] = ++stNodeId;
    // 线程2应处理的长度
    for(int i = 0; i < THREADNUM2; ++i)
		loopSet7_2[i] = loopSet[4][i];
    tdLen = nodeNum5; startId7[2] = stNodeId;
    for (; stNodeId < nodeNum; ++stNodeId) {
        size = nodeIntSize[4][stNodeId];
        if(size){
            loopSet[4][owners[stNodeId]] += size;
            tdLen += size;
            if (tdLen > blockPart) break;
        }
    }
    finishId7[2] = ++stNodeId;
    // 线程1应处理的长度
    for(int i = 0; i < THREADNUM2; ++i)
		loopSet7_1[i] = loopSet[4][i];
    tdLen = nodeNum4; startId7[1] = stNodeId;
    for (; stNodeId < nodeNum; ++stNodeId) {
        size = nodeIntSize[4][stNodeId];
        if(size){
            loopSet[4][owners[stNodeId]] += size;
            tdLen += size;
            if (tdLen > blockPart) break;
        }
    }
    finishId7[1] = ++stNodeId;
    // 线程0应处理的长度
	for (int i = 0; i < THREADNUM2; ++i)
		loopSet7_0[i] = loopSet[4][i];
    startId7[0] = stNodeId; finishId7[0] = nodeNum;

#ifdef TEST
    gettimeofday(&tv1, NULL);
    cout << "outTrans prepro clock is " << (tv1.tv_sec - tv0.tv_sec) * 1000 + (tv1.tv_usec - tv0.tv_usec) / 1000.0 << "ms\n";
    gettimeofday(&tv0, NULL);
#endif

    int charSize[4] = {0};    // 3456环字符数
    int charSize7[4] = {0};   // 7环分配给3456线程的buffer字符数
    thread t0(outTrans_mt, outbuffer3, loopSet, loopSet7_0, outbuffer73, startId7[0], finishId7[0], charSize, charSize7, 0);
	thread t1(outTrans_mt, outbuffer4, loopSet, loopSet7_1, outbuffer74, startId7[1], finishId7[1], charSize, charSize7, 1);
	thread t2(outTrans_mt, outbuffer5, loopSet, loopSet7_2, outbuffer75, startId7[2], finishId7[2], charSize, charSize7, 2);
	thread t3(outTrans_mt, outbuffer6, loopSet, loopSet7_3, outbuffer76, startId7[3], finishId7[3], charSize, charSize7, 3);
	t0.join();  t1.join(); t2.join(); t3.join();

#ifdef TEST
    gettimeofday(&tv1, NULL);
    cout << "outtrans clock is " << (tv1.tv_sec - tv0.tv_sec) + (tv1.tv_usec - tv0.tv_usec) / 1000000.0 << "s\n";
    gettimeofday(&tv0, NULL);
#endif
    // 写入文件
    string files = outputFiles;
	FILE* fp = fopen(files.c_str(), "w");
	if (fp == NULL) exit(0);

    fwrite(outbuffer3_0, 1, charSize[0]+intSize+1, fp);
    fwrite(outbuffer4, 1, charSize[1], fp);
    fwrite(outbuffer5, 1, charSize[2], fp);
    fwrite(outbuffer6, 1, charSize[3], fp);
    
    fwrite(outbuffer76, 1, charSize7[3], fp);
    fwrite(outbuffer75, 1, charSize7[2], fp);
    fwrite(outbuffer74, 1, charSize7[1], fp);
    fwrite(outbuffer73, 1, charSize7[0], fp);
    fclose(fp);
#ifdef TEST
    gettimeofday(&tv1, NULL);
    cout << "fwrite clock is " << (tv1.tv_sec - tv0.tv_sec) + (tv1.tv_usec - tv0.tv_usec) / 1000000.0 << "s\n";
#endif


}



int main()
{
#ifdef TEST
	struct timeval tv1, tv2, tv3, tv4, tv5;
	gettimeofday(&tv1, NULL);
#endif
	// 多线程读取数据
	loadData();		// 加载数据

#ifdef TEST
	struct timeval tv10, tv20;
	gettimeofday(&tv10, NULL);
	cout << "loadData : " << (tv10.tv_sec - tv1.tv_sec) + (tv10.tv_usec - tv1.tv_usec) / 1000000.0 << "s" << endl;
#endif
	// 建图
	buildGragh();

#ifdef TEST
	gettimeofday(&tv20, NULL);
	cout << "buildGraph : " << (tv20.tv_sec - tv10.tv_sec) + (tv20.tv_usec - tv10.tv_usec) / 1000000.0 << "s" << endl;
#endif

#ifdef TEST
	gettimeofday(&tv2, NULL);
	cout << "load data clock is : " << (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) / 1000000.0 << "s" << endl;
#endif
	applySpace();

	// 多线程找环	
	pthread_t td[THREADNUM2];
	for (int i = 0; i < THREADNUM2; i++) {
		pthread_create(&td[i], NULL, &findLoop_mt, thread_ids + i);
	}
    for (int i = 0; i < THREADNUM2; i++) {
		pthread_join(td[i], NULL);
	}

#ifdef TEST
	cout << "loopId:" << m_nodeNum << endl;
	gettimeofday(&tv3, NULL);
	cout << "find loop clock is : " << (tv3.tv_sec - tv2.tv_sec) + (tv3.tv_usec - tv2.tv_usec) / 1000000.0 << "s" << endl;
#endif
	// 按顺序取得各线程的环路字符
	outCharTrans();

#ifdef TEST
	gettimeofday(&tv4, NULL);
	cout << "out trans clock is : " << (tv4.tv_sec - tv3.tv_sec) + (tv4.tv_usec - tv3.tv_usec) / 1000000.0 << "s" << endl;
#endif

#ifdef TEST
	gettimeofday(&tv2, NULL);
	cout << "total clock is : " << (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) / 1000000.0 << "s" << endl;
#endif
}
