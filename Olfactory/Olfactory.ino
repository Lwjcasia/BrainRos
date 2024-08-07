volatile int MQ138;
volatile int MQ135;
volatile int MQ8;
volatile int MQ3;

void setup(){
  MQ138 = 0;
  MQ135 = 0;
  MQ8 = 0;
  MQ3 = 0;
  Serial.begin(9600);
}

void loop(){
  MQ138 = analogRead(A0);
  MQ135 = analogRead(A1);
  MQ8 = analogRead(A2);
  MQ3 = analogRead(A3);
  //Serial.write(MQ3);
  Serial.println(String("MQ138: ") + String(MQ138) + String("  ") + String("MQ135: ") + String(MQ135) + String("  ") + String("MQ8: ") + String(MQ8) + String("  ") + String("MQ3: ") + String(MQ3));
  //Serial.println(MQ8);
  delay(100);
  
}