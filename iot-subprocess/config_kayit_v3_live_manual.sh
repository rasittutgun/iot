#!/bin/bash
REC_NUM=100000 #alinacak kayıt sayisi
FREQ=2462e6 # alinan kaydın merkez frekansı (Ör: 10 MHz merkez için input: 10e6)
RATE=20e6 #kaydin örnekleme frekansı (Ör: 55 megasample için input : 55e6)
BW=22e6 # alinacak kaydin bant genişliği (Ör: 20 MHz için input : 20e6)
GAIN=40 # Cihazın gain'i
DUR=1 # bir kaydın alinacağı süre (saniye cinsinden)
NAME=iphone_3lu # oluşturulan kaydin ismi (Ör: deney_210912_...)
DIREC=/mnt/ssd1/dockerEndtoEndDemo/ahmet_test/ #kaydin oluşturulacağı dosya yolu
TECH=wif # hangi teknolojinin kaydinin alındığı (Ör: WiFi için "wif", Bluetooth için "ble")
TYPE=traf # hangi tipte sinyal kaydı alındığı (Ör: Hotspot içn "htsp", Trafik için "traf")
SLEEP=0.05 #iki kayit arası cihazın beklediği süre (saniye cinsinden)
SETUP=0.05 #cihazın setup süresi
CHZ=tel #Cihaz tipi (Ör:telefon,laptop)
MRK=Apple #Marka (Ör: Samsung,HP)
MDL=IPhone11 #Model (M31,IPhone6)
CHR= #Cihazın şarj yüzdesi (şarjlı bir cihaz değilse boş bırakılabilir)
PLG= #Cihaz şarja takılı mı "+" veya "-" şeklinde belirtilebilir (şarjlı bir cihaz değilse boş bırakılabilir)


bash record_kayit_v3_live.sh $REC_NUM $FREQ $RATE $BW $GAIN $DUR $NAME $DIREC $TECH $TYPE $SLEEP $SETUP $CHZ $MRK $MDL $CHR $PLG

exit 0
