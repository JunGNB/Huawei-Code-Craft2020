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

using namespace std;

// #define TEST

#define THREADNUM 8		// 线程数
#define THREADNUM2 8	// 找环线程数
#define deepMax 7
#define deepMin 2
#define DEGREE 64		// 入度出度的值
#define NODENUM 50000	// 最大节点ID以及节点个数
#define MAXNODE 50000 
#define lOOPNODEMAX 40000
#define sizeofchar 1
#define CHARLEN 8


#define LOOPSIZE3 1024 * 512 * 3 * 12 / 2	// 环路节点为3的环的字符总长度
#define LOOPSIZE4 1024 * 512 * 4 *12 / 2
#define LOOPSIZE5 1024 * 1024 * 5 * 12 / 2
#define LOOPSIZE6 1024 * 1024 * 6 * 12 / 2
#define LOOPSIZE7 1024 * 1024 * 2 * 7 * 12 / 2 

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#ifdef TEST
string inputFiles = "./test_data100w.txt";
string outputFiles = "./myresults3.txt";
#else
string inputFiles = "/data/test_data.txt";
string outputFiles = "/projects/student/result.txt";
#endif


struct Node {
	int* lastDataStart;		// 该节点的逆向邻接点的起始
	int* lastDataEnd;		// 该节点的逆向邻接点的终止
	int* nextDataStart;		// 该节点的正向邻接点的起始
	int* nextDataEnd;		// 该节点的正向邻接点的终止
};
int nodeCharSize[5][NODENUM];		// 存储每个节点的环的字符长度
int m_lastData[NODENUM * DEGREE];	// 存储节点的逆向邻接节点
int m_nextData[NODENUM * DEGREE];	// 存储节点的正向邻接节点
char m_chrSet[NODENUM][CHARLEN];	// 出现的节点ID字符串的集合
bool inGraph[NODENUM];				// 是否节点已经出现在图中的标志


// 存储字符环路的地址
char loopSetC3[THREADNUM2][LOOPSIZE3]; char loopSetC4[THREADNUM2][LOOPSIZE4]; char loopSetC5[THREADNUM2][LOOPSIZE5]; 
char loopSetC6[THREADNUM2][LOOPSIZE6]; char loopSetC7[THREADNUM2][LOOPSIZE7];
char outbuffer[1024 * 1024 * 2 * 7 * 11];


char iToch[10] = { 0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39 };
inline void intToStr(char* str, int num)
{
	char s[CHARLEN]; int i = 0, j = 0;

	s[i++] = num % 10 + 0x30;
	num /= 10;
	while (num > 0) {
		s[i++] = num % 10 + 0x30;
		num /= 10;
	}
	str[j++] = i + 1;	// 加上逗号
	while (i) {
		str[j++] = s[--i];
	}
	str[j++] = 0x2C;
}

struct loadDataParam {
	char* buf;
	int stbuf;
	int fnbuf;
	int* edge;
	int& edgeSize;
};

// 多线程加载文件，进行字符转换，存储到 edg 的图中， 第1个值存起始节点，第2个值存终止节点，依次存这两个值
void* loadData_mt(void* arg)
{
	struct loadDataParam* ldParams = (struct loadDataParam *)arg;
	char* buf = ldParams->buf;
	int stbuf = ldParams->stbuf;
	int fnbuf = ldParams->fnbuf;
	int* edge = ldParams->edge;
	int& edgeSize = ldParams->edgeSize;
	
	char* ch = NULL; 
	if (stbuf == 0) ch = buf;
	else {	// 找到当前行的起始位置作为起始点
		while (buf[stbuf--] != '\n');
		stbuf += 2;
		ch = &buf[stbuf];	// 下一行起始位置
	}
	
	while (buf[fnbuf--] != '\n');	// 找到当前行的起始位置作为终止点
	int bufSize = fnbuf + 2 - stbuf;

	int data0 = 0, data1 = 0;
	int fnSize = bufSize - 1;
	for (int i = 0; i < fnSize;) {
		while (ch[i] != ',') {	// 第一个字符
			data0 = (data0 << 1) + (data0 << 3) + (ch[i++] - '0');
		}
		i++; 

		while (ch[i] != ',') {	// 第二个字符
			data1 = (data1 << 1) + (data1 << 3) + (ch[i++] - '0');
		}
		if (data0 < NODENUM && data1 < NODENUM) {
			edge[edgeSize++] = data0;  edge[edgeSize++] = data1;
		}
		data0 = 0; data1= 0;
		i += 2; 

		while (ch[i++] != '\n'); // 第三个字符	
	}
}

// 数据载入
void loadData(vector<int>& sortId, int** edge, int* edgeSize, int& maxNode)
{
	int fd = open(inputFiles.c_str(), O_RDONLY);
	int size = lseek(fd, 0, SEEK_END);
	char* ch = (char*)mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
	close(fd);
	
	int part = size / THREADNUM;
	int edgsize_mt[THREADNUM] = {0}; 
	
	// 多线程数据载入
	pthread_t td[THREADNUM]; 
	vector<struct loadDataParam> ldParams; ldParams.reserve(THREADNUM);
	for (int i = 0; i < THREADNUM - 1; ++i) {
		struct loadDataParam ldp = {ch, i*part, (i+1)*part, edge[i], edgeSize[i]};
		ldParams.emplace_back(ldp);
		pthread_create(&td[i],NULL,&loadData_mt,&ldParams[i]);
	}
	struct loadDataParam ldp1 = {ch, (THREADNUM - 1)*part, size, edge[THREADNUM - 1], edgeSize[THREADNUM - 1]};
	pthread_create(&td[THREADNUM - 1], NULL, &loadData_mt, &ldp1);
	
	maxNode = MAXNODE;
	for (int i = 0; i < THREADNUM; ++i) {
		pthread_join(td[i], NULL);
	}
}

// 建图，因为数据加载弄了四个线程，因此，这里需要对四个 edge 的图进行遍历
void buildMmode(struct Node* m_node, vector<int>& sortId, int start, int end, const int* edge, const int edgeSize)
{
	
	int edgeEnd = edgeSize; int stNode, fnNode; 
	for (int i = 0; i < edgeEnd; i += 2) {
		if (edge[i] > start && edge[i] < end){		// 每个线程只对分块区域的节点进行操作
			if (inGraph[edge[i]]) {	// 头结点存在
				*m_node[edge[i]].nextDataEnd++ = edge[i + 1];
			}
			else
			{
				stNode = edge[i];
				m_node[stNode].lastDataStart = &m_lastData[stNode * DEGREE];
				m_node[stNode].lastDataEnd = m_node[stNode].lastDataStart;
				m_node[stNode].nextDataStart = &m_nextData[stNode * DEGREE];
				m_node[stNode].nextDataEnd = m_node[stNode].nextDataStart;
				*m_node[stNode].nextDataEnd++ = edge[i + 1];

				sortId.emplace_back(stNode);
				inGraph[stNode] = 1;
			}
		}
		
		if (edge[i+1] > start && edge[i+1] < end) {
			if (inGraph[edge[i + 1]]) {	// 尾结点存在
				*m_node[edge[i + 1]].lastDataEnd++ = edge[i];
			}
			else
			{
				fnNode = edge[i + 1];
				m_node[fnNode].lastDataStart = &m_lastData[fnNode * DEGREE];
				m_node[fnNode].lastDataEnd = m_node[fnNode].lastDataStart;
				m_node[fnNode].nextDataStart = &m_nextData[fnNode * DEGREE];
				m_node[fnNode].nextDataEnd = m_node[fnNode].nextDataStart;
				*m_node[fnNode].lastDataEnd++ = edge[i];
				sortId.emplace_back(fnNode);
				inGraph[fnNode] = 1;
			}
		}
	}
}

struct buildGraphParams {
	struct Node* m_node;
	vector<int>& sortId;
	int start;
	int end;
	int** edge;
	int* edgeSize;
};
// 多线程建图
void* buildGraph_mt(void* arg)
{
	struct buildGraphParams* bgp = (struct buildGraphParams*)arg;
	struct Node* m_node = bgp->m_node;
	vector<int>& sortId = bgp->sortId;
	int start = bgp->start;
	int end = bgp->end;
	int** edge = bgp->edge;
	int* edgeSize = bgp->edgeSize;
	
	// 对每个线程得到的边信息处理
	for(int i = 0; i < THREADNUM; ++i) {
		buildMmode(m_node, sortId, start, end, edge[i], edgeSize[i]);
	}
	
	sort(sortId.begin(), sortId.end());	// 部分节点排序
	auto stNode = sortId.begin(); auto fnNode = sortId.end(); 
	while(stNode < fnNode) {	
		if(m_node[*stNode].lastDataStart != m_node[*stNode].lastDataEnd) {	 // 有入度的值
			sort(m_node[*stNode].nextDataStart, m_node[*stNode].nextDataEnd);
			sort(m_node[*stNode].lastDataStart, m_node[*stNode].lastDataEnd);
			intToStr(m_chrSet[*stNode], *stNode);	// 存储节点对应的字符值
		}
		else {	// 无入度的节点
			m_node[*stNode].nextDataEnd = m_node[*stNode].nextDataStart;
		}
		++stNode;

	}
}


void buildGragh(vector<int>& sortId, struct Node* m_node, int* edge[THREADNUM], int* edgeSize, int maxNode)
{
	int part = maxNode / THREADNUM; vector<int> sortId_mt[THREADNUM - 1];
	pthread_t td[THREADNUM]; vector<struct buildGraphParams> bgp; bgp.reserve(THREADNUM);
	
	// 多线程建图
	struct buildGraphParams bp0 = {m_node, sortId,  -1, part, edge, edgeSize};
	pthread_create(&td[0], NULL, &buildGraph_mt, &bp0);
	
	for(int i = 1; i < THREADNUM - 1; i++) {
		sortId_mt[i-1].reserve(NODENUM / THREADNUM);
		struct buildGraphParams bp = {m_node, sortId_mt[i-1],  i*part - 1, (i+1)*part, edge, edgeSize};
		bgp.emplace_back(bp);
		pthread_create(&td[i], NULL, &buildGraph_mt, &bgp[i-1]);
	}
	
	struct buildGraphParams bpend = {m_node, sortId_mt[THREADNUM - 2], part * (THREADNUM - 1) - 1, maxNode + 1, edge, edgeSize};
	pthread_create(&td[THREADNUM - 1], NULL, &buildGraph_mt, &bpend);

	pthread_join(td[0], NULL);
	for(int i = 1; i < THREADNUM; i++) {
		pthread_join(td[i], NULL);
		sortId.insert(sortId.end(), sortId_mt[i - 1].begin(), sortId_mt[i - 1].end());	// 分块节点已经完成了排序，就不需要排序了
	}
}


// 多线程DFS
void DFS(int& stId, struct Node* m_node, char** loopSet, bitset<NODENUM>& adjNode, bitset<NODENUM>& visit, bool* instack)
{
	int loopV0, loopV1, loopV2, loopV3, loopV4, loopV5, loopV6;
	int stIdFlag = stId + 1;  int* lv2Node, *lv3Node, *lv4Node, *lv5Node, *lv6Node, *lv7Node;

	loopV0 = stId; 	// 第一层 
	lv2Node = upper_bound(m_node[stId].nextDataStart, m_node[stId].nextDataEnd, stId);
	for (; lv2Node != m_node[stId].nextDataEnd; ++lv2Node) {
		loopV1 = *lv2Node;
		instack[*lv2Node] = 1;

		lv3Node = upper_bound(m_node[*lv2Node].nextDataStart, m_node[*lv2Node].nextDataEnd, stId);
		for (; lv3Node != m_node[*lv2Node].nextDataEnd; ++lv3Node) {
			loopV2 = *lv3Node;  instack[*lv3Node] = 1; // 第三层
			if (adjNode[*lv3Node]) {	// 节点数3的环
				memcpy(loopSet[0], &m_chrSet[loopV0][1], m_chrSet[loopV0][0]);
				loopSet[0] += (int)m_chrSet[loopV0][0]; 
				memcpy(loopSet[0], &m_chrSet[loopV1][1], m_chrSet[loopV1][0]);
				loopSet[0] += (int)m_chrSet[loopV1][0]; 
				memcpy(loopSet[0], &m_chrSet[loopV2][1], m_chrSet[loopV2][0]);
				loopSet[0] += (int)m_chrSet[loopV2][0]; *(loopSet[0] - 1) = 0x0A; // 换行符
		
			}

			lv4Node = upper_bound(m_node[*lv3Node].nextDataStart, m_node[*lv3Node].nextDataEnd, stId);
			for (; lv4Node != m_node[*lv3Node].nextDataEnd; ++lv4Node) {
				if (!instack[*lv4Node]) {
					loopV3 = *lv4Node; instack[*lv4Node] = 1; // 第四层					
					if (adjNode[*lv4Node]) {	// 节点数4的环
						memcpy(loopSet[1], &m_chrSet[loopV0][1], m_chrSet[loopV0][0]);
						loopSet[1] += (int)m_chrSet[loopV0][0]; 
						memcpy(loopSet[1], &m_chrSet[loopV1][1], m_chrSet[loopV1][0]);
						loopSet[1] += (int)m_chrSet[loopV1][0]; 
						memcpy(loopSet[1], &m_chrSet[loopV2][1], m_chrSet[loopV2][0]);
						loopSet[1] += (int)m_chrSet[loopV2][0]; 
						memcpy(loopSet[1], &m_chrSet[loopV3][1], m_chrSet[loopV3][0]);
						loopSet[1] += (int)m_chrSet[loopV3][0]; *(loopSet[1] - 1) = 0x0A; // 换行符
					
					}

					lv5Node = upper_bound(m_node[*lv4Node].nextDataStart, m_node[*lv4Node].nextDataEnd, stId);
					for (; lv5Node != m_node[*lv4Node].nextDataEnd; ++lv5Node) {
						if (visit[*lv5Node] && !instack[*lv5Node]) {
							loopV4 = *lv5Node;  instack[*lv5Node] = 1; // 第五层							
							if (adjNode[*lv5Node]) {	// 节点数5的环
								memcpy(loopSet[2], &m_chrSet[loopV0][1], m_chrSet[loopV0][0]);
								loopSet[2] += (int)m_chrSet[loopV0][0]; 
								memcpy(loopSet[2], &m_chrSet[loopV1][1], m_chrSet[loopV1][0]);
								loopSet[2] += (int)m_chrSet[loopV1][0]; 
								memcpy(loopSet[2], &m_chrSet[loopV2][1], m_chrSet[loopV2][0]);
								loopSet[2] += (int)m_chrSet[loopV2][0]; 
								memcpy(loopSet[2], &m_chrSet[loopV3][1], m_chrSet[loopV3][0]);
								loopSet[2] += (int)m_chrSet[loopV3][0]; 
								memcpy(loopSet[2], &m_chrSet[loopV4][1], m_chrSet[loopV4][0]);
								loopSet[2] += (int)m_chrSet[loopV4][0]; *(loopSet[2] - 1) = 0x0A; // 换行符
							
							}

							lv6Node = upper_bound(m_node[*lv5Node].nextDataStart, m_node[*lv5Node].nextDataEnd, stId);
							for (; lv6Node != m_node[*lv5Node].nextDataEnd; ++lv6Node) {
								if (visit[*lv6Node] && !instack[*lv6Node]) {
									loopV5 = *lv6Node;
									if (adjNode[*lv6Node]) {	// 节点数6的环
										memcpy(loopSet[3], &m_chrSet[loopV0][1], m_chrSet[loopV0][0]);
										loopSet[3] += (int)m_chrSet[loopV0][0];
										memcpy(loopSet[3], &m_chrSet[loopV1][1], m_chrSet[loopV1][0]);
										loopSet[3] += (int)m_chrSet[loopV1][0]; 
										memcpy(loopSet[3], &m_chrSet[loopV2][1], m_chrSet[loopV2][0]);
										loopSet[3] += (int)m_chrSet[loopV2][0]; 
										memcpy(loopSet[3], &m_chrSet[loopV3][1], m_chrSet[loopV3][0]);
										loopSet[3] += (int)m_chrSet[loopV3][0]; 
										memcpy(loopSet[3], &m_chrSet[loopV4][1], m_chrSet[loopV4][0]);
										loopSet[3] += (int)m_chrSet[loopV4][0]; 
										memcpy(loopSet[3], &m_chrSet[loopV5][1], m_chrSet[loopV5][0]);
										loopSet[3] += (int)m_chrSet[loopV5][0]; *(loopSet[3] - 1) = 0x0A; // 换行符
									
									}

									lv7Node = upper_bound(m_node[*lv6Node].nextDataStart, m_node[*lv6Node].nextDataEnd, stId);
									for (; lv7Node != m_node[*lv6Node].nextDataEnd; ++lv7Node) {
										if (adjNode[*lv7Node] && !instack[*lv7Node]) { // 第七层，节点7的环
											loopV6 = *lv7Node;
											memcpy(loopSet[4], &m_chrSet[loopV0][1], m_chrSet[loopV0][0]);
											loopSet[4] += (int)m_chrSet[loopV0][0]; 
											memcpy(loopSet[4], &m_chrSet[loopV1][1], m_chrSet[loopV1][0]);
											loopSet[4] += (int)m_chrSet[loopV1][0]; 
											memcpy(loopSet[4], &m_chrSet[loopV2][1], m_chrSet[loopV2][0]);
											loopSet[4] += (int)m_chrSet[loopV2][0]; 
											memcpy(loopSet[4], &m_chrSet[loopV3][1], m_chrSet[loopV3][0]);
											loopSet[4] += (int)m_chrSet[loopV3][0]; 
											memcpy(loopSet[4], &m_chrSet[loopV4][1], m_chrSet[loopV4][0]);
											loopSet[4] += (int)m_chrSet[loopV4][0]; 
											memcpy(loopSet[4], &m_chrSet[loopV5][1], m_chrSet[loopV5][0]);
											loopSet[4] += (int)m_chrSet[loopV5][0]; 
											memcpy(loopSet[4], &m_chrSet[loopV6][1], m_chrSet[loopV6][0]);
											loopSet[4] += (int)m_chrSet[loopV6][0]; *(loopSet[4] - 1) = 0x0A; // 换行符
											
										}
									}
								}
							}
							instack[*lv5Node] = 0;
						}
					}
					instack[*lv4Node] = 0;
				}
			}
			instack[*lv3Node] = 0;
		}
		instack[*lv2Node] = 0;
	}
}

// BFS 遍历（确定5,6,7层可能出现的节点）
void BFS(int& fnId, struct Node* m_node, bitset<NODENUM>& adjNode, bitset<NODENUM>& visit, bool& flag)
{
	visit[fnId] = 1; int *midNode, *stNode, *lv5Node;
	midNode = upper_bound(m_node[fnId].lastDataStart, m_node[fnId].lastDataEnd, fnId);
	for (; midNode != m_node[fnId].lastDataEnd; ++midNode) {
		visit[*midNode] = 1;
		adjNode[*midNode] = 1;

		stNode = upper_bound(m_node[*midNode].lastDataStart, m_node[*midNode].lastDataEnd, fnId);
		for (; stNode != m_node[*midNode].lastDataEnd; ++stNode) {
			visit[*stNode] = 1;  flag = 1;
			lv5Node = upper_bound(m_node[*stNode].lastDataStart, m_node[*stNode].lastDataEnd, fnId);
			for (; lv5Node != m_node[*stNode].lastDataEnd; ++lv5Node) {
				visit[*lv5Node] = 1;
			}
		}
	}
}


struct findLoopPara
{
	struct Node* m_node;
	char** loopSet;
	vector<int>& sortId;
	int start;
	int tId;	// 线程号
};
void* findLoop_mt(void* arg)
{
	struct Node* m_node = ((struct findLoopPara*)arg)->m_node;
	char** loopSet = ((struct findLoopPara*)arg)->loopSet;
	vector<int>& sortId = ((struct findLoopPara*)arg)->sortId;
	int start = ((struct findLoopPara*)arg)->start;
	int tId = ((struct findLoopPara*)arg)->tId;
	
	int finish = (sortId.size() < lOOPNODEMAX) ? sortId.size() : lOOPNODEMAX;
	bool* instack = (bool*)calloc(NODENUM, sizeof(bool));		// 是否在栈中的标志
	bitset<NODENUM> visit;	
	bitset<NODENUM> adjNode;
	
	char *loop3Start, *loop4Start, *loop5Start, *loop6Start, *loop7Start; 
	int node;  bool flag = 0;// 是否逆向有环	
	// 直接找环
	for (int i = start; i < finish; i += THREADNUM2) {
		visit.reset();
		adjNode.reset();
		
		node = sortId[i];
		// bfs预处理		
		flag = 0;
		BFS(node, m_node, adjNode, visit, flag);
		// dfs开始遍历(逆向有第三层的路径)
		if (flag) {
			loop3Start = loopSet[0]; loop4Start = loopSet[1];  loop5Start = loopSet[2]; 
			loop6Start = loopSet[3]; loop7Start = loopSet[4]; 
			DFS(node, m_node, loopSet, adjNode, visit, instack);
			nodeCharSize[0][node] = (loopSet[0] - loop3Start);				// 存储该节点的环的总字符长度
			nodeCharSize[1][node] = (loopSet[1] - loop4Start);
			nodeCharSize[2][node] = (loopSet[2] - loop5Start);
			nodeCharSize[3][node] = (loopSet[3] - loop6Start);
			nodeCharSize[4][node] = (loopSet[4] - loop7Start);
		}
	}
}


inline void numToStr(char* str, int num, int& size)
{
	char s[10]; int i = 0, j = 0;

	s[i++] = iToch[num % 10];
	num /= 10;
	while (num > 0) {
		s[i++] = iToch[num % 10];
		num /= 10;
	}
	size = i;
	while (i) {
		str[j++] = s[--i];
	}
}

inline void strToInt(char* str, int& num) 
{
	int size = (int)*str++; num = 0;
	while(size--) {
		num = num * 10 + (*str++ - 0x30); 
	}
}

// 得到输出字符文件
void outCharTrans(vector<int>& sortId, int loopSum, int& bufSize)
{	
	bufSize = 0;  int intSize = 0;
#ifdef TEST
	cout << "符合规则的环有：" << loopSum << endl;
#endif
	// 第一行为环路个数
	char* outbufferPos = outbuffer;
	numToStr(outbufferPos, loopSum, intSize); outbufferPos += intSize;
	*outbufferPos++ = 0x0A;	// 换行符

	char* loopSet[THREADNUM2][5]; 
	for (int i = 0; i < THREADNUM2; i++) {
		loopSet[i][0] = loopSetC3[i]; loopSet[i][1] = loopSetC4[i]; loopSet[i][2] = loopSetC5[i];
		loopSet[i][3] = loopSetC6[i]; loopSet[i][4] = loopSetC7[i]; 	 	
	}

	char* loopPos[THREADNUM2];
	for (int i = 0; i < 5; ++i) {
		for (int j = 0; j < THREADNUM2; j++) {
			loopPos[j] = loopSet[j][i];
		}
		// nodeId 建立了映射关系，因此映射后的节点从0~sortId.size()-1;
		auto stNodeId = sortId.begin();auto stNodeEnd = sortId.end() - THREADNUM2; 
		int size = 0;
		for(; stNodeId < stNodeEnd;) {
			for (int k = 0; k < THREADNUM2; k++) {
				size = nodeCharSize[i][*stNodeId++];// 该节点无环，则得到的环为0，按顺序取节点的环即可。 
				if (size) {
					memcpy(outbufferPos, loopPos[k], size);
					outbufferPos += size;  loopPos[k] += size;
				}
			}
		}
		
		// 若节点数不是4的倍数（这里使用四线程），则最后几个需要判断是否节点已取完
		for (int k = 0; k < THREADNUM2; k++) {
			if (stNodeId < sortId.end()) { 
				if (size) {
					size = nodeCharSize[i][*stNodeId++];				
					memcpy(outbufferPos, loopPos[k], size);
					outbufferPos += size;  loopPos[k] += size;
				}
			}
		}
	}
	bufSize = (outbufferPos - outbuffer) / sizeof(char); 
}

void storeResults(int bufSize)
{
	string files = outputFiles;
	FILE* fp = fopen(files.c_str(), "w");
	if (fp == NULL) exit(0);

	int couter = fwrite(outbuffer, sizeof(char), bufSize, fp);
	fclose(fp);
}


int main()
{
#ifdef TEST
	struct timeval tv1, tv2, tv3, tv4, tv5;
	gettimeofday(&tv1, NULL);
#endif
	int bufSize = 0;
	vector<int> sortId; sortId.reserve(NODENUM);			// 数据文件中出现过得节点id
	memset(inGraph, 0, sizeof(bool) * NODENUM);
	memset(nodeCharSize, 0, sizeof(int) * NODENUM);
	
	// 多线程读取数据
	int *edg[THREADNUM];
	int edgSize[THREADNUM] = { 0 };
	for (int i = 0; i < THREADNUM; i++) {
		edg[i] = (int*)malloc(100 * 1024 * 2 * sizeof(int));// 读取数据边的集合
	}
	int maxNode = 0;
	loadData(sortId, edg, edgSize, maxNode);		// 加载数据
	
#ifdef TEST
	struct timeval tv10, tv20;
	gettimeofday(&tv10, NULL);
	cout << "loadData : " << (tv10.tv_sec - tv1.tv_sec) + (tv10.tv_usec - tv1.tv_usec) / 1000000.0 << "s" << endl;
#endif

	struct Node* m_node = (struct Node*)malloc(NODENUM * sizeof(struct Node));
	buildGragh(sortId, m_node, edg, edgSize, maxNode);		// 建图
#ifdef TEST
	gettimeofday(&tv20, NULL);
	cout << "buildGraph : " << (tv20.tv_sec - tv10.tv_sec) + (tv20.tv_usec - tv10.tv_usec) / 1000000.0 << "s" << endl;
#endif

#ifdef TEST
	gettimeofday(&tv2, NULL);
	cout << "load data clock is : " << (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) / 1000000.0 << "s" << endl;
#endif
	
	// 多线程找环	
	char* loopEnd[THREADNUM2][5];
	vector<struct findLoopPara> flp; flp.reserve(THREADNUM2);
	pthread_t t[THREADNUM2];
	for (int i = 0; i < THREADNUM2; i++) {
		loopEnd[i][0] = loopSetC3[i];  loopEnd[i][1] = loopSetC4[i];  loopEnd[i][2] = loopSetC5[i];
		loopEnd[i][3] = loopSetC6[i];  loopEnd[i][4] = loopSetC7[i];
		struct findLoopPara flp1 = {m_node, loopEnd[i], sortId, i, i};
		flp.emplace_back(flp1);
		pthread_create(&t[i], NULL, &findLoop_mt, &flp[i]);
	}
	
	int loopSum = 1004812;
	for (int i = 0; i < THREADNUM2; i++) {
		pthread_join(t[i], NULL);
	}
	
#ifdef TEST
	cout << "loopId:" << sortId.size() << endl;
	gettimeofday(&tv3, NULL);
	cout << "find loop clock is : " << (tv3.tv_sec - tv2.tv_sec) + (tv3.tv_usec - tv2.tv_usec) / 1000000.0 << "s" << endl;
#endif
	// 按顺序取得各线程的环路字符
	outCharTrans(sortId, loopSum, bufSize);

#ifdef TEST
	gettimeofday(&tv4, NULL);
	cout << "out trans clock is : " << (tv4.tv_sec - tv3.tv_sec) + (tv4.tv_usec - tv3.tv_usec) / 1000000.0 << "s" << endl;
#endif
	// 文件写入
	storeResults(bufSize);
#ifdef TEST
	gettimeofday(&tv5, NULL);
	cout << "write data clock is : " << (tv5.tv_sec - tv4.tv_sec) + (tv5.tv_usec - tv4.tv_usec) / 1000000.0 << "s" << endl;
#endif
	
#ifdef TEST
	gettimeofday(&tv2, NULL);
	cout << "total clock is : " << (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) / 1000000.0 << "s" << endl;
#endif
}
