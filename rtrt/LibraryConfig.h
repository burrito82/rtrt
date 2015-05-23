/*============================================================================*/
/*       1         2         3         4         5         6         7        */
/*3456789012345678901234567890123456789012345678901234567890123456789012345678*/
/*============================================================================*/
/*                                             .                              */
/*                                               RRRR WW  WW   WTTTTTTHH  HH  */
/*                                               RR RR WW WWW  W  TT  HH  HH  */
/*      Header   :                               RRRR   WWWWWWWW  TT  HHHHHH  */
/*                                               RR RR   WWW WWW  TT  HH  HH  */
/*      Module   :                               RR  R    WW  WW  TT  HH  HH  */
/*                                                                            */
/*      Project  :  Vista                          Rheinisch-Westfaelische    */
/*                                               Technische Hochschule Aachen */
/*      Purpose  :  DLL API                                                   */
/*                                                                            */
/*                                                 Copyright (c)  1998-2015   */
/*                                                 by  RWTH-Aachen, Germany   */
/*                                                 All rights reserved.       */
/*                                             .                              */
/*============================================================================*/
/*                                                                            */
/*    THIS SOFTWARE IS PROVIDED 'AS IS'. ANY WARRANTIES ARE DISCLAIMED. IN    */
/*    NO CASE SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES.    */
/*    REDISTRIBUTION AND USE OF THE NON MODIFIED TOOLKIT IS PERMITTED. OWN    */
/*    DEVELOPMENTS BASED ON THIS TOOLKIT MUST BE CLEARLY DECLARED AS SUCH.    */
/*                                                                            */
/*============================================================================*/
/*                                                                            */
/*      CLASS DEFINITIONS:                                                    */
/*                                                                            */
/*============================================================================*/

#ifndef RTRT_LIBRARYCONFIG_H
#define RTRT_LIBRARYCONFIG_H

// Windows DLL build
#if defined(WIN32) && !defined(rtrt_STATIC) 
#pragma warning(disable : 4251 4273 4275)
#ifdef rtrt_EXPORTS
#define RTRTAPI __declspec(dllexport)
#else
#define RTRTAPI __declspec(dllimport)
#endif
#else // no Windows or static build
#define RTRTAPI
#endif

#endif // ! RTRT_LIBRARYCONFIG_H

