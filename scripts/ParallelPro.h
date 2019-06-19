#ifndef PARALLEL_PRO_H
#define PARALLEL_PRO_H

class ParallelPro : public cv::ParallelLoopBody
{
private:
	cv::Mat m_img;
	cv::Mat &m_retVal;

public:
	ParallelPro(cv::Mat inputImage, cv::Mat &outImage) : m_img(inputImage), m_retVal(outImage)
	{

	}

	virtual void operator() (const cv::Range &range) const
	{
		for (int i = range.start; i != range.end; ++i)
		{

		}
	}
};

#endif
