// bls-app.cpp : This file contains the 'main' function. Program execution begins and ends there.
//


#include <cstdlib>
#include <iostream>
#include <boost/bind.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/asio.hpp>
#include <boost/thread/thread.hpp>
#include "BLS_Server.h"



int main(int argc, char* argv[])
{
	try
	{


		if (argc != 2)
		{
			std::cerr << "Usage: blocking_tcp_echo_server <port>\n";
			// return 1;
		}

		boost::asio::io_service io_service;

		//using namespace std; // For atoi.
		//server(io_service, atoi(argv[1]));
		server(io_service, port_num);


	}
	catch (std::exception& e)
	{
		std::cerr << "Exception: " << e.what() << "\n";
	}

	return 0;
}
