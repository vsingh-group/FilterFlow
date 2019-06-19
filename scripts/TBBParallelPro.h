#ifndef TBBPARALLEL_PRO_H
#define TBBPARALLEL_PRO_H

#include "blocked_range.h"
#include "omp.h"
using namespace tbb;

class TBBParallelPro
{
public:
	int m_arrayNum;

public:
	TBBParallelPro(int arrayNum) : m_arrayNum(arrayNum)
	{
	}
	void operator() (const blocked_range<int> &r) const
	{
#pragma  omp parallel 
		{
			for (int i = r.begin(); i != r.end(); ++i)
			{
				Foo(i);
			}
#pragma omp barrier
			for (int i = r.begin(); i != r.end(); ++i)
			{
				Foo2();
			}
		}
	}


private:
	void Foo(int i) const
	{
		if (i % 2 == 0)
		{
			printf("I am a scientist\n");
		}
		else
		{
			printf("I am a engineer\n");
		}
	}
	void Foo2() const
	{
		printf("I am a genius\n");
	}
};

#endif
