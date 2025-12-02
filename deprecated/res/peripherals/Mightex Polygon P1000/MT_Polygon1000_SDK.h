#pragma once
#ifndef _MT_Polygon1000_SDK_H_
#define _MT_Polygon1000_SDK_H_

typedef int SDK_RETURN_CODE;
typedef unsigned int DEV_HANDLE;


#ifdef SDK_EXPORTS
#define SDK_API extern "C" __declspec(dllexport) SDK_RETURN_CODE  _cdecl
#define SDK_HANDLE_API extern "C" __declspec(dllexport) DEV_HANDLE  _cdecl
#define SDK_POINTER_API extern "C" __declspec(dllexport) unsigned short*  _cdecl
#else
#define SDK_API extern "C" __declspec(dllimport) SDK_RETURN_CODE  _cdecl
#define SDK_HANDLE_API extern "C" __declspec(dllimport) DEV_HANDLE  _cdecl
#define SDK_POINTER_API extern "C" __declspec(dllimport) unsigned short*  _cdecl
#endif

#define DSI_IMGHEIGHT 1140
#define DSI_IMGWIDTH 912

typedef struct _imgSetting
{
	int VMirror = 0; //0 or  1, 0 for no mirror, 1 for apply mirror.
	int HMirror = 0; //
}tImgSetting;

typedef enum _ePtnTrigType
{
	PTT_CMD = 0,	//In pattern mode, pattern is displayed with command MTPLG_NextPattern
	PTT_AUTO = 1,	//In pattern mode, pattern is displayed automatically.
	PTT_EXT = 2,	//In pattern mode, pattern sequence is displayed with external trigger signal.
}ePtnTrigType;

typedef enum _eTrigEdgeType
{
	RISING_EDGE = 1,
	FALLING_EDGE = 2,
}eTrigEdgeType;

typedef struct _tPtnSetting
{
	int bitDepth = 1;	//bitDepth, currently we only support bitdepth = 1
	int ptnNum = 0;		//ptnNum
	int resverd2 = 0;	//reserved, DNU
	int ptntrigType = PTT_AUTO;//ePtnTrigType,which trigger ptn to run: possible are command, auto, external trigger,default to auto.
	int trigDelay = 0;	//input trigger delay in us, Current fixed to 0, do not changed.
	int resverd3 = 0;	//reserved, DNU 
	int expTime = 0;	//frametime in us
	int resverd4 = 0;	//reserved, DNU
	int trigType = 1;	//of eTrigEdgeType,defines input trigger edge type,default to RISING_EDGE
}tPtnSetting;

typedef struct _tTrigSetting
{
	int enabled = 1;   //Enabled, Output trigger is always enabled.
	int trigDelay = 0; //output trigger Delay in us, now fixed to 0, do not change.
	int trigPW = 100;  //trigger pulse width in us.
	int trigType = 1;  //of eTrigEdgeType,defines output triggers signal edge type,default to RISING_EDGE.
	int resverd3 = 0;  //reserved, DNU
}tTrigSetting;


typedef struct _deviceRunStatus
{
	int temperature = 0;	//device temperature
	int runState = 0;       //run state, possible values are 1:PARKED, 2:STOPPED, 3,RUNNING,4 PAUSED.
	int seqItemIdx = 0;     //pattern mode sequence index.
	int reserved = 0;
	int reserved1 = 0;
	int reserved2 = 0;
	int reserved3 = 0;
	int reserved4 = 0;
	int reserved5 = 0;
	int reserved6 = 0;
	int reserved7 = 0;
	int reserved8 = 0;
}tDevRunStatus;


//! MTPLG_InitDevice() - first function to call
//! Inputs: 
//!   devNum: -1 or actual device number[1,8]. -1 for automatic detect.
//! Return: > 0, device initiation successful with recoginized devices' number.
//!        = 0, no device is found.
//!        < 0, function call failed.  
SDK_API MTPLG_InitDevice(const int devNum = -1);

//! MTPLG_UnInitDevice - should be called after every MTPLG_InitDevice().
//! Inputs: - None
//! Return: always 0
SDK_API MTPLG_UnInitDevice();

//!! All devID is 1 based.

SDK_API MTPLG_ConnectDev(const int devID);
SDK_API MTPLG_DisconnectDev(const int devID);

//! MTPLG_GetDevModuleNo
//! Inputs: devID, 1 based device index.
//!        moduleNo[], of length 30.
SDK_API MTPLG_GetDevModuleNo(const int devID, char* moduleNo);

//! MTPLG_GetDevFMVers
//! Inputs:
SDK_API MTPLG_GetDevFMVers(const int devID, int* ver1, int* ver2, int* ver3);

//!MTPLG_SetImageSettings - global setting,affect all images uploaded to device after.
SDK_API MTPLG_SetImageSettings(const int devID, const tImgSetting aImgSetting);

//!MTPLG_SetDevDisplayMode
// Inputs: displayMode: 0-static mode; 1-patten mode
SDK_API MTPLG_SetDevDisplayMode(const int devID, const int displayMode);

//!MTPLG_SetDevStaticImageByFileName 
// Only supportted .bmp file with pixelformat = 1(monochromatic bitmap file).
SDK_API MTPLG_SetDevStaticImageByFileName(const int devID, const char* fileName);

//! MTPLG_SetDevStaticImageFromMemory
//! Inputs: pImgData - the pointer to the image pixel data(data only).
//!         imgPixelOrder - possible value is 0(for row major order) or 1(column major order).
//!             For Row Major Order, the bit data order will be pixel [column0, row0], [column1, row0],...[Column912,Row0];...
//!             For ColumnMajorOrder,the bit data order will be pixel [column0, row0],[column0, row1],...[Column1139][Row0];...
//!             The default is RowMajorOrder.
SDK_API MTPLG_SetDevStaticImageFromMemory(const int devID, unsigned char* pImgData, const int imgPixelOrder = 0);
SDK_API MTPLG_SetDevPtnSetting(const int devID, const tPtnSetting aPtnSetting);
SDK_API MTPLG_SetDevTrigSetting(const int devID, const tTrigSetting aTrigSetting);
SDK_API MTPLG_SetDevPtnByFileName(const int devID, const int ptnIdx, const char* fileName);
SDK_API MTPLG_SetDevPtnFromMemory(const int devID, const int ptnIdx, unsigned char* pImgData, const int imgPixelOrder = 0);
SDK_API MTPLG_StartPattern(const int devID);
SDK_API MTPLG_StopPattern(const int devID);
SDK_API MTPLG_NextPattern(const int devID);

//!MTPLG_UpdateDeviceFirmware
//! Inputs: fileName - absolute file path to the device firmware binary file(with extension .plb).
//!         timeOut_ms, in milliSeconds, due to different PC performance, the time needed for file transfer is different,
//!                     please leave enough time for this firmware update process.
//!                     or Set timeOut_ms to 0 and use MTPLG_GetDevUpdatedProgress to get the updating status.
SDK_API MTPLG_UpdateDeviceFirmware(const int devID, const char* fileName, const long int timeOut_ms = 600000);
SDK_API MTPLG_GetDevUpdateProgress(const int devID);
//!MTPLG_GetDevRunStatus
//! Acquire the device current status, including temperature, current pattern index when in pattern mode.
SDK_API MTPLG_GetDevRunStatus(const int devID, tDevRunStatus* pDevStatus, const long int timeOut_ms = -1);

#endif

