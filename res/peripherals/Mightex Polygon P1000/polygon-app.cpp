//
// blocking_tcp_echo_server.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2016 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <cstdlib>
#include <iostream>
#include <boost/bind.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/asio.hpp>
#include <boost/thread/thread.hpp>
#include "PolygonServer.h"
#include "MT_Polygon1000_SDK.h"


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