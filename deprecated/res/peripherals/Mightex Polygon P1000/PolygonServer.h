#pragma once
#include <cstdlib>
#include <iostream>
#include <boost/bind.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/asio.hpp>
#include <boost/thread/thread.hpp>
#include "MT_Polygon1000_SDK.h"
#include <bitset>
#include <chrono>
#include <ctime>  
#include <iomanip>
#include <sstream>
#include <string>

using boost::asio::ip::tcp;

// constants
const int port_num = 5007;
const int max_length = 1024;
const unsigned long MAX_IMAGEHEIGHT = DSI_IMGHEIGHT;
const unsigned long MAX_IMAGEWIDTH = DSI_IMGWIDTH;
const unsigned long MAX_IMAGESIZE = MAX_IMAGEHEIGHT * MAX_IMAGEWIDTH;

typedef boost::shared_ptr<tcp::socket> socket_ptr;

// Functions for interacting with polygon devices
//!Example 1 Auto InitDevices ==
void AutoInitDevice()
{
	// look for any connected devices
	//int aFlag = MTPLG_InitDevice(-1);
	//std::cout << "Auto InitDevice: " << aFlag << std::endl;
	//std::cout << "max image height: " << MAX_IMAGEHEIGHT << ", max image width: " << MAX_IMAGEWIDTH << ", max size: " << MAX_IMAGESIZE << std::endl;

	//! Get first device's SN
	int ret;
	int DevHandle = 1;
	char moduleNo[31] = {0};

	//initialize device
	std::cout << "Initializing Polygon device " << DevHandle;
	int aFlag = MTPLG_InitDevice(DevHandle);
	std::cout << ", result: " << aFlag << std::endl;

	// get module number
	std::cout << "Grabbing module number of Polygon device " << DevHandle;
	ret = MTPLG_GetDevModuleNo(DevHandle, moduleNo);
	std::cout << ", result: " << ret << ":" << moduleNo << std::endl;

	//! connect first device
	std::cout << "Connecting to Polygon device " << DevHandle;
	ret = MTPLG_ConnectDev(DevHandle);
	std::cout << ", result: " << ret << std::endl;

	// set display mode to static
	std::cout << "Setting Polygon device " << DevHandle << " to static display mode";
	ret = MTPLG_SetDevDisplayMode(1, 0);
	std::cout << ", result: " << ret << std::endl;

	//};
}

// function to change the dev index
void DisplayMask(const int devIdx, unsigned char *pImgData)
{

	// upload to device
	const int imgPixelOrder = 0;
	MTPLG_SetDevStaticImageFromMemory(devIdx, pImgData, imgPixelOrder);
};

std::string time_in_HH_MM_SS_MMM()
{
	using namespace std::chrono;

	// get current time
	auto now = system_clock::now();

	// get number of milliseconds for the current second
	// (remainder after division into seconds)
	auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

	// convert to std::time_t in order to convert to std::tm (broken time)
	auto timer = system_clock::to_time_t(now);

	// convert to broken time
	std::tm bt{};
	localtime_s(&bt, &timer);

	std::ostringstream oss;

	oss << std::put_time(&bt, "%H:%M:%S"); // HH:MM:SS
	oss << '.' << std::setfill('0') << std::setw(3) << ms.count();

	return oss.str();
}

// server functions
void session(socket_ptr sock)
{
	try
	{
		for (;;)
		{

			// initialize buffer of appropriate size
			unsigned char *data = new unsigned char[MAX_IMAGESIZE];
			//char data[max_length];

			// create boost buffer
			boost::asio::mutable_buffer buff = boost::asio::buffer(data, MAX_IMAGESIZE);

			// read socket into boost buffer
			boost::system::error_code error;
			//size_t length = sock->read_some(boost::asio::buffer(data), error);
			size_t length = sock->read_some(buff, error);
			if (error == boost::asio::error::eof)
				break; // Connection closed cleanly by peer.
			else if (error)
				throw boost::system::system_error(error); // Some other error.

			// print what we received
			//std::cout << "data: " << data << std::endl;
			//std::cout << "sizeof(data): " << sizeof(data) << std::endl;
			//std::cout << "length: " << length << std::endl;
			//std::cout << "buff: " << buff.data() << std::endl;

			//std::string now = time_in_HH_MM_SS_MMM();
			//std::cout << "Polygon sending command at: " << now << std::endl;

			// upload to device
			DisplayMask(1, data);

			// write back useful information like timestamp
			//boost::asio::write(*sock, boost::asio::buffer(data, length));
			//std::string message = "cool";
			//boost::asio::write(*sock, boost::asio::buffer(message));
			
			// get current timestamp
			//auto now = std::chrono::system_clock::now();
			//std::time_t end_time = std::chrono::system_clock::to_time_t(now);
			//std::string later = time_in_HH_MM_SS_MMM();
			//std::cout << "Polygon command completed: " << later << std::endl;

		}
	}
	catch (std::exception &e)
	{
		std::cerr << "Exception in thread: " << e.what() << "\n";

		// handle exception/shutdown
		// this gets thrown if client side closes the socket
		MTPLG_UnInitDevice();
	}
}

void server(boost::asio::io_service &io_service, unsigned short port)
{
	tcp::acceptor a(io_service, tcp::endpoint(tcp::v4(), port));
	for (;;)
	{
		socket_ptr sock(new tcp::socket(io_service));
		a.accept(*sock);
		std::cout << "Accepted socket connection" << std::endl;

		// after socket connection gets established
		AutoInitDevice();

		// put reading from socket on its own thread
		boost::thread t(boost::bind(session, sock));
	}
}

