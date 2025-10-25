typedef int SDK_RETURN_CODE;

#ifdef SDK_EXPORTS
#define SDK_API extern "C" __declspec(dllexport) SDK_RETURN_CODE _cdecl
#else
#define SDK_API extern "C" __declspec(dllimport) SDK_RETURN_CODE _cdecl
#endif

#define  MAX_PULSE_COUNT     21 

#define DISABLE_MODE	0
#define NORMAL_MODE		1
#define TRIGGER_MODE    3

SDK_API MTUSB_BLSDriverInitDevices(void);
SDK_API MTUSB_BLSDriverOpenDevice(int DeviceIndex);
SDK_API MTUSB_BLSDriverCloseDevice(int DevHandle);
SDK_API MTUSB_BLSDriverGetSerialNo(int DevHandle, unsigned char* SerNumber, int Size);
SDK_API MTUSB_BLSDriverGetModuleType(int DevHandle);
SDK_API MTUSB_BLSDriverGetChannels(int DevHandle);
SDK_API MTUSB_BLSDriverGetChannelTitle(int DevHandle, int ChnlIdx, unsigned char* ChnlTitle);
SDK_API MTUSB_BLSDriverSetMode(int DevHandle, int Channel, int Mode);
SDK_API MTUSB_BLSDriverSetNormalCurrent(int DevHandle, int Channel, int Current);
SDK_API MTUSB_BLSDriverSetPulseProfile(int DevHandle, int Channel, int Polarity, int PulseCnt, int ReptCnt);
SDK_API MTUSB_BLSDriverSetPulseDetail(int DevHandle, int Channel, int PulseIndex, int Time0, int Time1, int Time2, int Curr0, int Curr1, int Curr2);
SDK_API MTUSB_BLSDriverSetFollowModeDetail(int DevHanlde, int Channel, int ION, int IOFF);
SDK_API MTUSB_BLSDriverSoftStart(int DevHandle, int Channel);
SDK_API MTUSB_BLSDriverResetDevice(int DevHandle);
SDK_API MTUSB_BLSDriverStorePara(int DevHandle);
SDK_API MTUSB_BLSDriverGetEPSMCount(int DevHandle, int Channel);
SDK_API MTUSB_BLSDriverSendCommand(int DevHandle, char* Command);
