#pragma once
#include <cstdlib>
#include <iostream>
#include <boost/bind.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/asio.hpp>
#include <boost/thread/thread.hpp>
#include "Mightex_BLSDriver_SDK.h"
#include <chrono>
#include <ctime>  
#include <iomanip>
#include <sstream>
#include <string>

using boost::asio::ip::tcp;

// constants
const int port_num = 5008;
const int max_length = 1024;

typedef boost::shared_ptr<tcp::socket> socket_ptr;

// function to take input from python and respond accordingly
void TriggerLight(int DevHandle, int channel, int value)
{

	int ret;
	std::cout << "BLSDriverSetNormalCurrent channel " << channel << " to " << value;
	ret = MTUSB_BLSDriverSetNormalCurrent(DevHandle, channel, value);
	std::cout << ", ret: " << ret << std::endl;
}

// Functions for interacting with polygon devices
//!Example 1 Auto InitDevices ==
int AutoInitDevice()
{

	// general garbage return value holder
	int ret;

	// scan for connected devices
	int USBDevices;
	USBDevices = MTUSB_BLSDriverInitDevices();
	std::cout << "Number of BLS devices connected: " << USBDevices << std::endl;

	// open device
	int DevHandle = -1;
	DevHandle = MTUSB_BLSDriverOpenDevice(0);
	if (DevHandle == -1)
	{
		std::cout << " Device Open Failed! " << std::endl;
	}

	// get serial number
	unsigned char SerialNum[32];
	ret = MTUSB_BLSDriverGetSerialNo(DevHandle, SerialNum, 32);
	std::cout << "BLS Device, DevHandle " << DevHandle << ", SerialNum: " << SerialNum << std::endl;

	// get channel number
	int channels;
	channels = MTUSB_BLSDriverGetChannels(DevHandle);
	std::cout << "BLS Channels: " << channels << std::endl;

	// get channel names
	unsigned char ChnlTitle[64];
	int length;
	std::cout << "Grabbing channel names..." << std::endl;
	for (int i = 0; i < channels; i++)
	{

		// get channel titles, 1-indexed
		length = MTUSB_BLSDriverGetChannelTitle(DevHandle, i + 1, ChnlTitle);
		std::cout << "Channel " << i + 1 << ": " << ChnlTitle << std::endl;
	};

	// set channel mode, start with normal (0=disable, 1=normal)
	for (int i = 0; i < channels; i++)
	{
		ret = MTUSB_BLSDriverSetMode(DevHandle, i + 1, 1);
	};

	// set current (start with off), in 0.1% unit (so 1000 means max)
	for (int i = 0; i < channels; i++)
	{
		ret = MTUSB_BLSDriverSetNormalCurrent(DevHandle, i + 1, 0);
	};

	return DevHandle;
}

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

		// garbage collection var
		int ret;

		// initialize connection to device
		int DevHandle = -1;
		DevHandle = AutoInitDevice();

		for (;;)
		{

			// initialize buffer of appropriate size
			int buffsize = 1024;
			int *data = new int[1024];

			// create boost buffer
			boost::asio::mutable_buffer buff = boost::asio::buffer(data, buffsize);

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
			//std::cout << "BLS sending command at: " << now << std::endl;

			// respond to data input
			int channel = data[0];
			int value = data[1];
			TriggerLight(DevHandle, channel, value);

			// write back useful information like timestamp
			//boost::asio::write(*sock, boost::asio::buffer(data, length));
			//std::string message = "thx";
			//boost::asio::write(*sock, boost::asio::buffer(message));

			// get timestamp
			//std::string later = time_in_HH_MM_SS_MMM();
			//std::cout << "BLS command finished at: " << later << std::endl;
		}
	}
	catch (std::exception &e)
	{
		std::cerr << "Exception in thread: " << e.what() << "\n";

		// handle exception/shutdown
		// this gets thrown if client side closes the socket
		int ret;

		// hopefully this takes care of everything we need it to
		std::cout << "Closing devices..." << std::endl;
		int DevHandle = 0;
		ret = MTUSB_BLSDriverCloseDevice(DevHandle);
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

		// put reading from socket on its own thread
		boost::thread t(boost::bind(session, sock));
	}
}
