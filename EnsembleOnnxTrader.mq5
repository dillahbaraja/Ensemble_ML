//+------------------------------------------------------------------+
//|                                            EnsembleOnnxTrader.mq5 |
//|  Loads multiple ONNX classifiers, applies weighted voting, and   |
//|  executes MT5 trades from the exported DataExporter feature set. |
//+------------------------------------------------------------------+
#property strict
#property version   "1.00"

#include <Trade/Trade.mqh>

enum ENUM_SIGNAL_SOURCE
  {
   SIGNAL_LR = 0,
   SIGNAL_SVM = 1,
   SIGNAL_RF = 2,
   SIGNAL_XGB = 3,
   SIGNAL_ENSEMBLE = 4
  };

input string InpModelFolder = "EnsembleOnnxTrader";
input ENUM_SIGNAL_SOURCE InpSignalSource = SIGNAL_ENSEMBLE;
input double InpLots = 0.10;
input int InpSlPoints = 300;
input int InpTpPoints = 600;
input bool InpUseCommonFiles = true;
input bool InpUseCpuOnly = true;
input bool InpCloseOnHold = false;
input ulong InpMagicNumber = 26072401;
input int InpCooldownBars = 12;
input double InpMinSignalEdge = 0.03;
input int InpMinAgreementVotes = 2;
input bool InpEnableTransactionCsv = true;
input string InpTransactionCsvFile = "";
input bool InpExportHistoryOnDeinit = true;
input bool InpUseNewsFilter = true;
input string InpNewsCsvFile = "Assets\\MT5_2018.01.01-2026.06.30_ALL_.csv";
input int InpNewsMinImpact = 0;

CTrade trade;

string gModelNames[] = {"lr", "svm", "rf", "xgb"};
int gModelCount = 4;
long gOnnxHandles[];
double gModelWeights[];
string gModelFiles[];
double gModelThresholds[];

string gFeatureNames[];
double gFeatureMeans[];
double gFeatureStds[];
int gFeatureCount = 0;
double gThreshold = 0.45;
int gClassCount = 3;

int handleRSI = INVALID_HANDLE;
int handleATR = INVALID_HANDLE;
int handleMACD = INVALID_HANDLE;
int handleBands = INVALID_HANDLE;
int handleStoch = INVALID_HANDLE;
int handleADX = INVALID_HANDLE;
int handleRSI_H4 = INVALID_HANDLE;
int handleMA_H4 = INVALID_HANDLE;
int handleRSI_M15 = INVALID_HANDLE;
int handleATR_M15 = INVALID_HANDLE;

datetime gNewsEventTimes[];
int gNewsEventSignals[];
string gNewsEventNames[];
string gNewsEventCountryCodes[];
double gNewsEventActuals[];
double gNewsEventForecasts[];
double gNewsEventPreviouss[];
int gNewsEventImpacts[];
int gNewsEventCount = 0;

datetime gLastBarTime = 0;
string gTransactionCsvFile = "";
string gResolvedConfigFile = "";
string gResolvedSignalSource = "";
int gBarsToWaitBeforeEntry = 0;
string gCurrentNewsSummary = "NEWS:none";
string gCurrentNewsJournal = "NEWS:none";
string gCurrentNewsComment = "NEWS:none";
string gCurrentNewsSelectedName = "";
string gCurrentNewsSelectedCountry = "";
double gCurrentNewsSelectedActual = 0.0;
double gCurrentNewsSelectedForecast = 0.0;
double gCurrentNewsSelectedPrevious = 0.0;
int gCurrentNewsSelectedImpact = 0;
int gCurrentNewsSelectedSignal = 0;
int gCurrentNewsMatchedCount = 0;

bool LoadConfig();
bool LoadModels();
void ReleaseModels();
bool SetupIndicators();
void ReleaseIndicators();
bool LoadNewsEvents();
int ComputeNewsSignal(const datetime previous_bar_time, const datetime current_bar_time);
int NewsPairSideSign(const string country_code);
bool PrepareNewsContext(const datetime previous_bar_time, const datetime current_bar_time, const int signal);
string BuildNewsEventText(const int event_index);
string BuildTradeComment(const int signal);
string NewsSignalToText(const int signal);
bool GetFeatureValue(const string feature_name, const int shift, double &value);
bool BuildFeatureVector(vectorf &features);
bool RunModel(const int model_index, const vectorf &features, double &prob_down, double &prob_hold, double &prob_up);
bool PassLongIndicatorFilter();
bool PassShortIndicatorFilter();
int ComputeEnsembleSignal(const vectorf &features, double &bull_prob, double &bear_prob);
bool ProcessSignal(const int signal, const int news_signal);
bool HasOpenPosition(const long position_type);
bool CloseOpenPositions();
string BuildModelPath(const string file_name);
bool ReadIndicatorValue(const int handle, const int buffer_index, const int shift, double &value);
string DealTypeToText(const ENUM_DEAL_TYPE deal_type);
string DealEntryToText(const ENUM_DEAL_ENTRY deal_entry);
bool AppendDealCsv(const ulong deal_ticket);
bool ExportAllDealHistoryCsv();
string ResolveTransactionCsvFile();
string ResolveConfigFileName();
string ResolveSignalSourceName();
void ApplyCooldown();
double Clamp01(const double value);

//+------------------------------------------------------------------+
int OnInit()
  {
   ArrayResize(gOnnxHandles, gModelCount);
   ArrayResize(gModelWeights, gModelCount);
   ArrayResize(gModelFiles, gModelCount);
   ArrayResize(gModelThresholds, gModelCount);

   trade.SetExpertMagicNumber(InpMagicNumber);
   gResolvedConfigFile = ResolveConfigFileName();
   if(StringLen(gResolvedConfigFile) == 0)
     {
      Print("Unsupported chart symbol for config mapping: ", _Symbol);
      return INIT_FAILED;
     }
   gResolvedSignalSource = ResolveSignalSourceName();
   gTransactionCsvFile = ResolveTransactionCsvFile();

   if(!LoadConfig())
      return INIT_FAILED;

   if(!SetupIndicators())
      return INIT_FAILED;

   if(InpUseNewsFilter && !LoadNewsEvents())
      return INIT_FAILED;

   if(!LoadModels())
      return INIT_FAILED;

   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(InpEnableTransactionCsv && InpExportHistoryOnDeinit)
      ExportAllDealHistoryCsv();
   ReleaseModels();
   ReleaseIndicators();
  }

//+------------------------------------------------------------------+
void OnTick()
  {
   MqlRates rates[2];
   if(CopyRates(_Symbol, PERIOD_H1, 0, 2, rates) < 2)
      return;

   if(rates[1].time == gLastBarTime)
      return;

   gLastBarTime = rates[1].time;
   ApplyCooldown();

   vectorf features(gFeatureCount);
   if(!BuildFeatureVector(features))
      return;

   double bull_prob = 0.0;
   double bear_prob = 0.0;
   int signal = ComputeEnsembleSignal(features, bull_prob, bear_prob);
   int news_signal = 0;
   if(InpUseNewsFilter)
      news_signal = ComputeNewsSignal(rates[1].time, rates[0].time);
   PrepareNewsContext(rates[1].time, rates[0].time, signal);
   PrintFormat("%s signal=%d news=%d bull=%.4f bear=%.4f", gResolvedSignalSource, signal, news_signal, bull_prob, bear_prob);

   ProcessSignal(signal, news_signal);
  }

//+------------------------------------------------------------------+
bool LoadConfig()
  {
   int flags = FILE_READ | FILE_CSV | FILE_ANSI;
   if(InpUseCommonFiles)
      flags |= FILE_COMMON;

   string config_path = InpModelFolder + "\\" + gResolvedConfigFile;
   int handle = FileOpen(config_path, flags, ',');
   if(handle == INVALID_HANDLE)
     {
      Print("Failed to open config file: ", config_path, " err=", GetLastError());
      return false;
     }

   bool header = true;
   while(!FileIsEnding(handle))
     {
      string row_type = FileReadString(handle);
      string key = FileReadString(handle);
      string value1 = FileReadString(handle);
      string value2 = FileReadString(handle);
      string value3 = FileReadString(handle);

      if(header)
        {
         header = false;
         continue;
        }

      if(row_type == "PARAM" && (key == "ensemble_threshold" || key == "threshold"))
         gThreshold = StringToDouble(value1);
      else if(row_type == "PARAM" && key == "class_count")
        {
         gClassCount = (int)StringToInteger(value1);
         if(gClassCount < 2)
            gClassCount = 3;
        }
      else if(row_type == "WEIGHT")
        {
         for(int i = 0; i < gModelCount; i++)
           {
            if(gModelNames[i] == key)
              {
               gModelWeights[i] = StringToDouble(value1);
               break;
              }
            }
        }
      else if(row_type == "THRESHOLD")
        {
         for(int i = 0; i < gModelCount; i++)
           {
            if(gModelNames[i] == key)
              {
               gModelThresholds[i] = StringToDouble(value1);
               break;
              }
           }
        }
      else if(row_type == "MODEL")
        {
         for(int i = 0; i < gModelCount; i++)
           {
            if(gModelNames[i] == key)
              {
               gModelFiles[i] = value1;
               break;
              }
           }
        }
      else if(row_type == "FEATURE")
        {
         int idx = (int)StringToInteger(key);
         if(idx + 1 > gFeatureCount)
           {
            gFeatureCount = idx + 1;
            ArrayResize(gFeatureNames, gFeatureCount);
            ArrayResize(gFeatureMeans, gFeatureCount);
            ArrayResize(gFeatureStds, gFeatureCount);
           }
         gFeatureNames[idx] = value1;
         gFeatureMeans[idx] = StringToDouble(value2);
         gFeatureStds[idx] = StringToDouble(value3);
         if(gFeatureStds[idx] == 0.0)
            gFeatureStds[idx] = 1.0;
        }
     }

   FileClose(handle);
   if(gFeatureCount <= 0)
     {
      Print("No features loaded from config.");
      return false;
     }
   return true;
  }

//+------------------------------------------------------------------+
bool LoadModels()
  {
   uint onnx_flags = ONNX_COMMON_FOLDER;
   if(!InpUseCommonFiles)
      onnx_flags = 0;
   if(InpUseCpuOnly)
      onnx_flags |= ONNX_USE_CPU_ONLY;

   ulong output0[] = {1};
   ulong output1[] = {1, (ulong)gClassCount};

   for(int i = 0; i < gModelCount; i++)
     {
      string model_path = BuildModelPath(gModelFiles[i]);
      gOnnxHandles[i] = OnnxCreate(model_path, onnx_flags);
      if(gOnnxHandles[i] == INVALID_HANDLE)
        {
         Print("Failed to load ONNX model: ", model_path, " err=", GetLastError());
         return false;
        }

      ulong shaped_input[] = {1, (ulong)gFeatureCount};
      if(!OnnxSetInputShape(gOnnxHandles[i], 0, shaped_input))
        {
         Print("Failed to set input shape for ", model_path, " err=", GetLastError());
         return false;
        }
      if(!OnnxSetOutputShape(gOnnxHandles[i], 0, output0))
        {
         Print("Failed to set label output shape for ", model_path, " err=", GetLastError());
         return false;
        }
      if(!OnnxSetOutputShape(gOnnxHandles[i], 1, output1))
        {
         Print("Failed to set probability output shape for ", model_path, " err=", GetLastError());
         return false;
        }
     }

   return true;
  }

//+------------------------------------------------------------------+
void ReleaseModels()
  {
   for(int i = 0; i < ArraySize(gOnnxHandles); i++)
     {
      if(gOnnxHandles[i] != INVALID_HANDLE)
        {
         OnnxRelease(gOnnxHandles[i]);
         gOnnxHandles[i] = INVALID_HANDLE;
        }
     }
  }

//+------------------------------------------------------------------+
bool SetupIndicators()
  {
   handleRSI = iRSI(_Symbol, PERIOD_H1, 14, PRICE_CLOSE);
   handleATR = iATR(_Symbol, PERIOD_H1, 14);
   handleMACD = iMACD(_Symbol, PERIOD_H1, 12, 26, 9, PRICE_CLOSE);
   handleBands = iBands(_Symbol, PERIOD_H1, 20, 0, 2.0, PRICE_CLOSE);
   handleStoch = iStochastic(_Symbol, PERIOD_H1, 5, 3, 3, MODE_SMA, STO_LOWHIGH);
   handleADX = iADX(_Symbol, PERIOD_H1, 14);
   handleRSI_H4 = iRSI(_Symbol, PERIOD_H4, 14, PRICE_CLOSE);
   handleMA_H4 = iMA(_Symbol, PERIOD_H4, 24, 0, MODE_SMA, PRICE_CLOSE);
   handleRSI_M15 = iRSI(_Symbol, PERIOD_M15, 14, PRICE_CLOSE);
   handleATR_M15 = iATR(_Symbol, PERIOD_M15, 14);

   if(handleRSI == INVALID_HANDLE || handleATR == INVALID_HANDLE ||
      handleMACD == INVALID_HANDLE || handleBands == INVALID_HANDLE ||
      handleStoch == INVALID_HANDLE || handleADX == INVALID_HANDLE ||
      handleRSI_H4 == INVALID_HANDLE || handleMA_H4 == INVALID_HANDLE ||
      handleRSI_M15 == INVALID_HANDLE || handleATR_M15 == INVALID_HANDLE)
     {
      Print("Indicator initialization failed. err=", GetLastError());
      return false;
     }
   return true;
  }

//+------------------------------------------------------------------+
void ReleaseIndicators()
  {
   if(handleRSI != INVALID_HANDLE) IndicatorRelease(handleRSI);
   if(handleATR != INVALID_HANDLE) IndicatorRelease(handleATR);
   if(handleMACD != INVALID_HANDLE) IndicatorRelease(handleMACD);
   if(handleBands != INVALID_HANDLE) IndicatorRelease(handleBands);
   if(handleStoch != INVALID_HANDLE) IndicatorRelease(handleStoch);
   if(handleADX != INVALID_HANDLE) IndicatorRelease(handleADX);
   if(handleRSI_H4 != INVALID_HANDLE) IndicatorRelease(handleRSI_H4);
   if(handleMA_H4 != INVALID_HANDLE) IndicatorRelease(handleMA_H4);
   if(handleRSI_M15 != INVALID_HANDLE) IndicatorRelease(handleRSI_M15);
   if(handleATR_M15 != INVALID_HANDLE) IndicatorRelease(handleATR_M15);
  }

//+------------------------------------------------------------------+
bool LoadNewsEvents()
  {
   int flags = FILE_READ | FILE_CSV | FILE_ANSI;
   if(InpUseCommonFiles)
      flags |= FILE_COMMON;

   int handle = FileOpen(InpNewsCsvFile, flags, ',');
   if(handle == INVALID_HANDLE)
     {
      Print("Failed to open news file: ", InpNewsCsvFile, " err=", GetLastError());
      return false;
     }

   ArrayResize(gNewsEventTimes, 0);
   ArrayResize(gNewsEventSignals, 0);
   ArrayResize(gNewsEventNames, 0);
   ArrayResize(gNewsEventCountryCodes, 0);
   ArrayResize(gNewsEventActuals, 0);
   ArrayResize(gNewsEventForecasts, 0);
   ArrayResize(gNewsEventPreviouss, 0);
   ArrayResize(gNewsEventImpacts, 0);
   gNewsEventCount = 0;

   bool header = true;
   while(!FileIsEnding(handle))
     {
      string id = FileReadString(handle);
      string datetime_str = FileReadString(handle);
      string time_str = FileReadString(handle);
      string name = FileReadString(handle);
      string country_code = FileReadString(handle);
      string country_name = FileReadString(handle);
      string importance_str = FileReadString(handle);
      string actual_str = FileReadString(handle);
      string forecast_str = FileReadString(handle);
      string previous_str = FileReadString(handle);
      string impact_str = FileReadString(handle);
      string url = FileReadString(handle);

      if(header)
        {
         header = false;
         continue;
        }

      datetime event_time = StringToTime(datetime_str);
      if(event_time <= 0)
         continue;

      int importance = (int)StringToInteger(importance_str);

      double actual = StringToDouble(actual_str);
      double forecast = StringToDouble(forecast_str);
      if(actual == 0.0 && forecast == 0.0)
         continue;

      int pair_side = NewsPairSideSign(country_code);
      if(pair_side == 0)
         continue;

      double delta = actual - forecast;
      int event_signal = 0;
      if(delta > 0.0)
         event_signal = pair_side;
      else if(delta < 0.0)
         event_signal = -pair_side;

      if(event_signal == 0)
         continue;

      int new_size = gNewsEventCount + 1;
      ArrayResize(gNewsEventTimes, new_size);
      ArrayResize(gNewsEventSignals, new_size);
      ArrayResize(gNewsEventNames, new_size);
      ArrayResize(gNewsEventCountryCodes, new_size);
      ArrayResize(gNewsEventActuals, new_size);
      ArrayResize(gNewsEventForecasts, new_size);
      ArrayResize(gNewsEventPreviouss, new_size);
      ArrayResize(gNewsEventImpacts, new_size);
      gNewsEventTimes[gNewsEventCount] = event_time;
      gNewsEventSignals[gNewsEventCount] = event_signal;
      gNewsEventNames[gNewsEventCount] = name;
      gNewsEventCountryCodes[gNewsEventCount] = country_code;
      gNewsEventActuals[gNewsEventCount] = actual;
      gNewsEventForecasts[gNewsEventCount] = forecast;
      gNewsEventPreviouss[gNewsEventCount] = StringToDouble(previous_str);
      gNewsEventImpacts[gNewsEventCount] = importance;
      gNewsEventCount = new_size;
     }

   FileClose(handle);
   Print("Loaded ", gNewsEventCount, " news filter events from ", InpNewsCsvFile);
   return true;
  }

//+------------------------------------------------------------------+
int NewsPairSideSign(const string country_code)
  {
   string symbol = _Symbol;
   StringToUpper(symbol);
   string code = country_code;
   StringToUpper(code);

   if(symbol == "EURUSD")
     {
      if(code == "EU")
         return 1;
      if(code == "US")
         return -1;
     }
   else if(symbol == "GBPUSD")
     {
      if(code == "GB")
         return 1;
      if(code == "US")
         return -1;
     }
   else if(symbol == "EURJPY")
     {
      if(code == "EU")
         return 1;
     }
   else if(symbol == "USDJPY")
     {
      if(code == "US")
         return 1;
     }

   return 0;
  }

//+------------------------------------------------------------------+
int ComputeNewsSignal(const datetime previous_bar_time, const datetime current_bar_time)
  {
   if(gNewsEventCount <= 0)
      return 0;

   int bull_votes = 0;
   int bear_votes = 0;

   for(int i = 0; i < gNewsEventCount; i++)
     {
      if(gNewsEventTimes[i] >= previous_bar_time && gNewsEventTimes[i] < current_bar_time)
        {
         if(gNewsEventSignals[i] > 0)
            bull_votes++;
         else if(gNewsEventSignals[i] < 0)
            bear_votes++;
        }
     }

   if(bull_votes > bear_votes)
      return 1;
   if(bear_votes > bull_votes)
      return -1;
   return 0;
  }

//+------------------------------------------------------------------+
string NewsSignalToText(const int signal)
  {
   if(signal > 0)
      return "BULL";
   if(signal < 0)
      return "BEAR";
   return "NONE";
  }

//+------------------------------------------------------------------+
string BuildNewsEventText(const int event_index)
  {
   if(event_index < 0 || event_index >= gNewsEventCount)
      return "";

   string name = gNewsEventNames[event_index];
   StringReplace(name, ",", " ");
   StringReplace(name, ";", " ");

   string code = gNewsEventCountryCodes[event_index];
   StringToUpper(code);

   return StringFormat("%s[%s] I=%d A=%s F=%s P=%s",
                       name,
                       code,
                       gNewsEventImpacts[event_index],
                       DoubleToString(gNewsEventActuals[event_index], 4),
                       DoubleToString(gNewsEventForecasts[event_index], 4),
                       DoubleToString(gNewsEventPreviouss[event_index], 4));
  }

//+------------------------------------------------------------------+
string BuildTradeComment(const int signal)
  {
   string side = "HOLD";
   if(signal > 0)
      side = "BUY";
   else if(signal < 0)
      side = "SELL";

   string comment = StringFormat("%s_%s_%s", gResolvedSignalSource, side, gCurrentNewsComment);
   StringReplace(comment, " ", "_");
   if(StringLen(comment) > 60)
      comment = StringSubstr(comment, 0, 60);
   return comment;
  }

//+------------------------------------------------------------------+
bool PrepareNewsContext(const datetime previous_bar_time, const datetime current_bar_time, const int signal)
  {
   gCurrentNewsSummary = "NEWS:none";
   gCurrentNewsJournal = "NEWS:none";
   gCurrentNewsComment = "NEWS:none";
   gCurrentNewsSelectedName = "";
   gCurrentNewsSelectedCountry = "";
   gCurrentNewsSelectedActual = 0.0;
   gCurrentNewsSelectedForecast = 0.0;
   gCurrentNewsSelectedPrevious = 0.0;
   gCurrentNewsSelectedImpact = 0;
   gCurrentNewsSelectedSignal = 0;
   gCurrentNewsMatchedCount = 0;

   if(!InpUseNewsFilter || gNewsEventCount <= 0)
      return true;

   int best_idx = -1;
   double best_score = -1.0;
   string all_events = "";

   for(int i = 0; i < gNewsEventCount; i++)
     {
      if(gNewsEventTimes[i] < previous_bar_time || gNewsEventTimes[i] >= current_bar_time)
         continue;

      gCurrentNewsMatchedCount++;

      string event_text = BuildNewsEventText(i);
      if(StringLen(all_events) > 0)
         all_events += " | ";
      all_events += event_text;

      int event_signal = gNewsEventSignals[i];
      double delta = MathAbs(gNewsEventActuals[i] - gNewsEventForecasts[i]);
      double align_bonus = 0.0;
      if(signal != 0 && event_signal == signal)
         align_bonus = 100000.0;
      double score = align_bonus + ((double)gNewsEventImpacts[i] * 1000.0) + (delta * 10.0);
      if(score > best_score)
        {
         best_score = score;
         best_idx = i;
        }
     }

   if(best_idx >= 0)
     {
      gCurrentNewsSelectedName = gNewsEventNames[best_idx];
      gCurrentNewsSelectedCountry = gNewsEventCountryCodes[best_idx];
      gCurrentNewsSelectedActual = gNewsEventActuals[best_idx];
      gCurrentNewsSelectedForecast = gNewsEventForecasts[best_idx];
      gCurrentNewsSelectedPrevious = gNewsEventPreviouss[best_idx];
      gCurrentNewsSelectedImpact = gNewsEventImpacts[best_idx];
      gCurrentNewsSelectedSignal = gNewsEventSignals[best_idx];

      string selected_text = BuildNewsEventText(best_idx);
      gCurrentNewsSummary = StringFormat("NEWS[%d] %s", gCurrentNewsMatchedCount, selected_text);
      gCurrentNewsJournal = StringFormat("NEWS window=%s signal=%s matched=%d selected=%s all=%s",
                                         TimeToString(previous_bar_time, TIME_DATE | TIME_MINUTES),
                                         NewsSignalToText(signal),
                                         gCurrentNewsMatchedCount,
                                         selected_text,
                                         all_events);

      string comment_text = StringFormat("%s %s", NewsSignalToText(signal), selected_text);
      StringReplace(comment_text, " ", "_");
      if(StringLen(comment_text) > 60)
         comment_text = StringSubstr(comment_text, 0, 60);
      gCurrentNewsComment = comment_text;
     }
   else
     {
      gCurrentNewsSummary = "NEWS:none";
      gCurrentNewsJournal = StringFormat("NEWS window=%s signal=%s matched=0",
                                         TimeToString(previous_bar_time, TIME_DATE | TIME_MINUTES),
                                         NewsSignalToText(signal));
      gCurrentNewsComment = "NEWS:none";
     }

   return true;
  }

//+------------------------------------------------------------------+
bool GetFeatureValue(const string feature_name, const int shift, double &value)
  {
   MqlRates rate[1];
   if(feature_name == "Open" || feature_name == "High" || feature_name == "Low" ||
      feature_name == "Close" || feature_name == "Volume" || feature_name == "Hour" ||
      feature_name == "DayOfWeek")
     {
      if(CopyRates(_Symbol, PERIOD_H1, shift, 1, rate) < 1)
         return false;

      if(feature_name == "Open") value = rate[0].open;
      else if(feature_name == "High") value = rate[0].high;
      else if(feature_name == "Low") value = rate[0].low;
      else if(feature_name == "Close") value = rate[0].close;
      else if(feature_name == "Volume") value = (double)rate[0].tick_volume;
      else
        {
         MqlDateTime dt;
         TimeToStruct(rate[0].time, dt);
         if(feature_name == "Hour")
            value = (double)dt.hour;
         else
            value = (double)dt.day_of_week;
        }
      return true;
     }

   double buffer[1];
   if(feature_name == "RSI")
      return ReadIndicatorValue(handleRSI, 0, shift, value);
   if(feature_name == "ATR")
      return ReadIndicatorValue(handleATR, 0, shift, value);
   if(feature_name == "MACD_Main")
      return ReadIndicatorValue(handleMACD, 0, shift, value);
   if(feature_name == "MACD_Signal")
      return ReadIndicatorValue(handleMACD, 1, shift, value);
   if(feature_name == "Bands_Upper")
      return ReadIndicatorValue(handleBands, 1, shift, value);
   if(feature_name == "Bands_Lower")
      return ReadIndicatorValue(handleBands, 2, shift, value);
   if(feature_name == "Stoch_Main")
      return ReadIndicatorValue(handleStoch, 0, shift, value);
   if(feature_name == "Stoch_Signal")
      return ReadIndicatorValue(handleStoch, 1, shift, value);
   if(feature_name == "ADX_Main")
      return ReadIndicatorValue(handleADX, 0, shift, value);
   if(feature_name == "ADX_PlusDI")
      return ReadIndicatorValue(handleADX, 1, shift, value);
   if(feature_name == "ADX_MinusDI")
      return ReadIndicatorValue(handleADX, 2, shift, value);

   MqlRates bar[1];
   if(CopyRates(_Symbol, PERIOD_H1, shift, 1, bar) < 1)
      return false;

   int h4_shift = iBarShift(_Symbol, PERIOD_H4, bar[0].time, false);
   int m15_shift = iBarShift(_Symbol, PERIOD_M15, bar[0].time, false);
   if(h4_shift < 0 || m15_shift < 0)
      return false;

   if(feature_name == "RSI_H4")
      return ReadIndicatorValue(handleRSI_H4, 0, h4_shift, value);
   if(feature_name == "MA_H4")
      return ReadIndicatorValue(handleMA_H4, 0, h4_shift, value);
   if(feature_name == "RSI_M15")
      return ReadIndicatorValue(handleRSI_M15, 0, m15_shift, value);
   if(feature_name == "ATR_M15")
      return ReadIndicatorValue(handleATR_M15, 0, m15_shift, value);

   value = 0.0;
   return true;
  }

//+------------------------------------------------------------------+
bool BuildFeatureVector(vectorf &features)
  {
   int feature_total = ArraySize(gFeatureNames);
   for(int i = 0; i < feature_total; i++)
     {
      double raw = 0.0;
      if(!GetFeatureValue(gFeatureNames[i], 1, raw))
        {
         Print("Failed to read feature: ", gFeatureNames[i]);
         return false;
        }
      features[i] = (float)((raw - gFeatureMeans[i]) / gFeatureStds[i]);
     }
   return true;
  }

//+------------------------------------------------------------------+
bool RunModel(const int model_index, const vectorf &features, double &prob_down, double &prob_hold, double &prob_up)
  {
   vectorf label_output(1);
   vectorf prob_output(gClassCount);
   vectorf input_copy = features;

   if(!OnnxRun(gOnnxHandles[model_index], ONNX_DEFAULT, input_copy, label_output, prob_output))
     {
        Print("OnnxRun failed for model ", gModelNames[model_index], " err=", GetLastError());
        return false;
     }

   if(gClassCount >= 3)
     {
      prob_down = Clamp01(prob_output[0]);
      prob_hold = Clamp01(prob_output[1]);
      prob_up = Clamp01(prob_output[2]);
     }
   else
     {
      prob_down = Clamp01(prob_output[0]);
      prob_hold = 0.0;
      prob_up = Clamp01(prob_output[1]);
     }
   return true;
  }

//+------------------------------------------------------------------+
bool PassLongIndicatorFilter()
  {
   double close = 0.0, ma_h4 = 0.0, rsi = 0.0, rsi_h4 = 0.0, rsi_m15 = 0.0;
   double macd_main = 0.0, macd_signal = 0.0, stoch_main = 0.0, stoch_signal = 0.0;
   double adx_main = 0.0, adx_plus = 0.0, adx_minus = 0.0;

   if(!GetFeatureValue("Close", 1, close) ||
      !GetFeatureValue("MA_H4", 1, ma_h4) ||
      !GetFeatureValue("RSI", 1, rsi) ||
      !GetFeatureValue("RSI_H4", 1, rsi_h4) ||
      !GetFeatureValue("RSI_M15", 1, rsi_m15) ||
      !GetFeatureValue("MACD_Main", 1, macd_main) ||
      !GetFeatureValue("MACD_Signal", 1, macd_signal) ||
      !GetFeatureValue("Stoch_Main", 1, stoch_main) ||
      !GetFeatureValue("Stoch_Signal", 1, stoch_signal) ||
      !GetFeatureValue("ADX_Main", 1, adx_main) ||
      !GetFeatureValue("ADX_PlusDI", 1, adx_plus) ||
      !GetFeatureValue("ADX_MinusDI", 1, adx_minus))
      return false;

   int score = 0;
   score += (close > ma_h4) ? 1 : 0;
   score += (rsi >= 50.0 && rsi <= 75.0) ? 1 : 0;
   score += (rsi_h4 >= 50.0) ? 1 : 0;
   score += (rsi_m15 >= 50.0) ? 1 : 0;
   score += (macd_main > macd_signal) ? 1 : 0;
   score += (stoch_main > stoch_signal) ? 1 : 0;
   score += (adx_main >= 15.0 && adx_plus > adx_minus) ? 1 : 0;
   return score >= 4;
  }

//+------------------------------------------------------------------+
bool PassShortIndicatorFilter()
  {
   double close = 0.0, ma_h4 = 0.0, rsi = 0.0, rsi_h4 = 0.0, rsi_m15 = 0.0;
   double macd_main = 0.0, macd_signal = 0.0, stoch_main = 0.0, stoch_signal = 0.0;
   double adx_main = 0.0, adx_plus = 0.0, adx_minus = 0.0;

   if(!GetFeatureValue("Close", 1, close) ||
      !GetFeatureValue("MA_H4", 1, ma_h4) ||
      !GetFeatureValue("RSI", 1, rsi) ||
      !GetFeatureValue("RSI_H4", 1, rsi_h4) ||
      !GetFeatureValue("RSI_M15", 1, rsi_m15) ||
      !GetFeatureValue("MACD_Main", 1, macd_main) ||
      !GetFeatureValue("MACD_Signal", 1, macd_signal) ||
      !GetFeatureValue("Stoch_Main", 1, stoch_main) ||
      !GetFeatureValue("Stoch_Signal", 1, stoch_signal) ||
      !GetFeatureValue("ADX_Main", 1, adx_main) ||
      !GetFeatureValue("ADX_PlusDI", 1, adx_plus) ||
      !GetFeatureValue("ADX_MinusDI", 1, adx_minus))
      return false;

   int score = 0;
   score += (close < ma_h4) ? 1 : 0;
   score += (rsi >= 25.0 && rsi <= 50.0) ? 1 : 0;
   score += (rsi_h4 <= 50.0) ? 1 : 0;
   score += (rsi_m15 <= 50.0) ? 1 : 0;
   score += (macd_main < macd_signal) ? 1 : 0;
   score += (stoch_main < stoch_signal) ? 1 : 0;
   score += (adx_main >= 15.0 && adx_minus > adx_plus) ? 1 : 0;
   return score >= 4;
  }

//+------------------------------------------------------------------+
int ComputeEnsembleSignal(const vectorf &features, double &bull_prob, double &bear_prob)
  {
   double ensemble_down = 0.0;
   double ensemble_hold = 0.0;
   double ensemble_up = 0.0;

   if(InpSignalSource != SIGNAL_ENSEMBLE)
     {
      int model_index = (int)InpSignalSource;
      double prob_down = 0.5;
      double prob_hold = 0.0;
      double prob_up = 0.5;
      double model_threshold = gModelThresholds[model_index];
      if(model_threshold <= 0.0)
         model_threshold = gThreshold;
      if(model_index < 0 || model_index >= gModelCount || !RunModel(model_index, features, prob_down, prob_hold, prob_up))
        {
         bull_prob = 0.5;
         bear_prob = 0.5;
         return 0;
        }

      bull_prob = Clamp01(prob_up);
      bear_prob = Clamp01(prob_down);
      if(prob_up >= model_threshold && prob_up >= prob_hold && prob_up >= prob_down && (prob_up - MathMax(prob_hold, prob_down)) >= InpMinSignalEdge)
         return 1;
      if(prob_down >= model_threshold && prob_down >= prob_hold && prob_down >= prob_up && (prob_down - MathMax(prob_hold, prob_up)) >= InpMinSignalEdge)
         return -1;
      if(prob_up > prob_down && (prob_up - prob_down) >= InpMinSignalEdge && prob_up >= 0.35)
         return 1;
      if(prob_down > prob_up && (prob_down - prob_up) >= InpMinSignalEdge && prob_down >= 0.35)
         return -1;
      return 0;
     }

   double bull_score = 0.0;
   double bear_score = 0.0;
   double hold_score = 0.0;
   int bull_votes = 0;
   int bear_votes = 0;

   for(int i = 0; i < gModelCount; i++)
     {
      double prob_down = 0.5;
      double prob_hold = 0.0;
      double prob_up = 0.5;
      if(!RunModel(i, features, prob_down, prob_hold, prob_up))
         return 0;

      double model_threshold = gModelThresholds[i];
      if(model_threshold <= 0.0)
         model_threshold = gThreshold;

      ensemble_down += gModelWeights[i] * prob_down;
      ensemble_hold += gModelWeights[i] * prob_hold;
      ensemble_up += gModelWeights[i] * prob_up;

      if(prob_up >= model_threshold && prob_up >= prob_hold && prob_up >= prob_down && (prob_up - MathMax(prob_hold, prob_down)) >= InpMinSignalEdge)
        {
         bull_score += gModelWeights[i];
         bull_votes++;
        }
      else if(prob_down >= model_threshold && prob_down >= prob_hold && prob_down >= prob_up && (prob_down - MathMax(prob_hold, prob_up)) >= InpMinSignalEdge)
        {
         bear_score += gModelWeights[i];
         bear_votes++;
        }
      else
         hold_score += gModelWeights[i];
     }

   double total = bull_score + bear_score + hold_score;
   if(total <= 0.0)
     {
      bull_prob = 0.5;
      bear_prob = 0.5;
      return 0;
     }

   double weight_total = 0.0;
   for(int j = 0; j < gModelCount; j++)
      weight_total += gModelWeights[j];
   if(weight_total <= 0.0)
      weight_total = 1.0;

   ensemble_down = Clamp01(ensemble_down / weight_total);
   ensemble_hold = Clamp01(ensemble_hold / weight_total);
   ensemble_up = Clamp01(ensemble_up / weight_total);
   bull_prob = ensemble_up;
   bear_prob = ensemble_down;

   if(bull_votes >= InpMinAgreementVotes && ensemble_up >= gThreshold && ensemble_up >= ensemble_hold && ensemble_up >= ensemble_down && (ensemble_up - MathMax(ensemble_hold, ensemble_down)) >= InpMinSignalEdge)
      return 1;
   if(bear_votes >= InpMinAgreementVotes && ensemble_down >= gThreshold && ensemble_down >= ensemble_hold && ensemble_down >= ensemble_up && (ensemble_down - MathMax(ensemble_hold, ensemble_up)) >= InpMinSignalEdge)
      return -1;
   if(ensemble_up > ensemble_down && (ensemble_up - ensemble_down) >= InpMinSignalEdge && ensemble_up >= 0.35)
      return 1;
   if(ensemble_down > ensemble_up && (ensemble_down - ensemble_up) >= InpMinSignalEdge && ensemble_down >= 0.35)
      return -1;
   return 0;
  }

//+------------------------------------------------------------------+
bool ProcessSignal(const int signal, const int news_signal)
  {
   bool has_buy = HasOpenPosition(POSITION_TYPE_BUY);
   bool has_sell = HasOpenPosition(POSITION_TYPE_SELL);
   string order_comment = BuildTradeComment(signal);

   if(signal == 1)
     {
      if(InpUseNewsFilter && news_signal < 0)
         return true;
      if(!PassLongIndicatorFilter())
         return true;
      if(gBarsToWaitBeforeEntry > 0)
         return true;
      if(has_sell && !CloseOpenPositions())
         return false;
      if(!has_buy)
        {
         double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         double sl = (InpSlPoints > 0) ? ask - InpSlPoints * _Point : 0.0;
         double tp = (InpTpPoints > 0) ? ask + InpTpPoints * _Point : 0.0;
         PrintFormat("%s BUY | %s | %s", gResolvedSignalSource, gCurrentNewsSummary, gCurrentNewsJournal);
         bool ok = trade.Buy(InpLots, _Symbol, ask, sl, tp, order_comment);
         if(ok)
            gBarsToWaitBeforeEntry = MathMax(InpCooldownBars, 0);
         return ok;
        }
      return true;
     }

   if(signal == -1)
     {
      if(InpUseNewsFilter && news_signal > 0)
         return true;
      if(!PassShortIndicatorFilter())
         return true;
      if(gBarsToWaitBeforeEntry > 0)
         return true;
      if(has_buy && !CloseOpenPositions())
         return false;
      if(!has_sell)
        {
         double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         double sl = (InpSlPoints > 0) ? bid + InpSlPoints * _Point : 0.0;
         double tp = (InpTpPoints > 0) ? bid - InpTpPoints * _Point : 0.0;
         PrintFormat("%s SELL | %s | %s", gResolvedSignalSource, gCurrentNewsSummary, gCurrentNewsJournal);
         bool ok = trade.Sell(InpLots, _Symbol, bid, sl, tp, order_comment);
         if(ok)
            gBarsToWaitBeforeEntry = MathMax(InpCooldownBars, 0);
         return ok;
        }
      return true;
     }

   if(InpCloseOnHold && (has_buy || has_sell))
     {
      bool ok = CloseOpenPositions();
      if(ok)
         gBarsToWaitBeforeEntry = MathMax(InpCooldownBars, 0);
      return ok;
     }
   return true;
  }

//+------------------------------------------------------------------+
bool HasOpenPosition(const long position_type)
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;
      if(!PositionSelectByTicket(ticket))
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;
      if(PositionGetInteger(POSITION_TYPE) == position_type)
         return true;
     }
   return false;
  }

//+------------------------------------------------------------------+
bool CloseOpenPositions()
  {
   bool ok = true;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;
      if(!PositionSelectByTicket(ticket))
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;
      if(!trade.PositionClose(ticket))
         ok = false;
     }
   return ok;
  }

//+------------------------------------------------------------------+
string BuildModelPath(const string file_name)
  {
   return InpModelFolder + "\\" + file_name;
  }

//+------------------------------------------------------------------+
bool ReadIndicatorValue(const int handle, const int buffer_index, const int shift, double &value)
  {
   double buffer[1];
   if(CopyBuffer(handle, buffer_index, shift, 1, buffer) < 1)
      return false;
   value = buffer[0];
   return true;
  }

//+------------------------------------------------------------------+
void ApplyCooldown()
  {
   if(gBarsToWaitBeforeEntry > 0)
      gBarsToWaitBeforeEntry--;
  }

//+------------------------------------------------------------------+
double Clamp01(const double value)
  {
   if(value < 0.0)
      return 0.0;
   if(value > 1.0)
      return 1.0;
   return value;
  }

//+------------------------------------------------------------------+
string ResolveTransactionCsvFile()
  {
   string trimmed = InpTransactionCsvFile;
   StringTrimLeft(trimmed);
   StringTrimRight(trimmed);
   if(StringLen(trimmed) > 0)
      return trimmed;

   string config_stem = gResolvedConfigFile;
   StringToLower(config_stem);
   StringReplace(config_stem, ".csv", "");
   return "EnsembleOnnxTrader_" + config_stem + "_" + gResolvedSignalSource + "_transactions.csv";
  }

//+------------------------------------------------------------------+
string ResolveConfigFileName()
  {
   string symbol = _Symbol;
   StringToUpper(symbol);

   if(symbol == "EURUSD")
      return "ensemble_config_eurusd_h1.csv";
   if(symbol == "GBPUSD")
      return "ensemble_config_gbpusd_h1.csv";
   if(symbol == "AUDUSD")
      return "ensemble_config_audusd_h1.csv";
   if(symbol == "EURJPY")
      return "ensemble_config_eurjpy_h1.csv";
   if(symbol == "USDJPY")
      return "ensemble_config_usdjpy_h1.csv";

   return "";
  }

//+------------------------------------------------------------------+
string ResolveSignalSourceName()
  {
   switch(InpSignalSource)
     {
      case SIGNAL_LR:
         return "lr";
      case SIGNAL_SVM:
         return "svm";
      case SIGNAL_RF:
         return "rf";
      case SIGNAL_XGB:
         return "xgb";
      case SIGNAL_ENSEMBLE:
         return "ensemble";
     }

   return "ensemble";
  }

//+------------------------------------------------------------------+
string DealTypeToText(const ENUM_DEAL_TYPE deal_type)
  {
   if(deal_type == DEAL_TYPE_BUY)
      return "BUY";
   if(deal_type == DEAL_TYPE_SELL)
      return "SELL";
   return "OTHER";
  }

//+------------------------------------------------------------------+
string DealEntryToText(const ENUM_DEAL_ENTRY deal_entry)
  {
   if(deal_entry == DEAL_ENTRY_IN)
      return "IN";
   if(deal_entry == DEAL_ENTRY_OUT)
      return "OUT";
   if(deal_entry == DEAL_ENTRY_INOUT)
      return "INOUT";
   if(deal_entry == DEAL_ENTRY_OUT_BY)
      return "OUT_BY";
   return "OTHER";
  }

//+------------------------------------------------------------------+
bool AppendDealCsv(const ulong deal_ticket)
  {
   if(!InpEnableTransactionCsv || deal_ticket == 0)
      return false;
   if(!HistoryDealSelect(deal_ticket))
      return false;

   string symbol = HistoryDealGetString(deal_ticket, DEAL_SYMBOL);
   if(symbol != _Symbol)
      return true;
   if((ulong)HistoryDealGetInteger(deal_ticket, DEAL_MAGIC) != InpMagicNumber)
      return true;

   int flags = FILE_READ | FILE_WRITE | FILE_CSV | FILE_ANSI;
   if(InpUseCommonFiles)
      flags |= FILE_COMMON;

   int fh = FileOpen(gTransactionCsvFile, flags, ',');
   if(fh == INVALID_HANDLE)
     {
      flags = FILE_WRITE | FILE_CSV | FILE_ANSI;
      if(InpUseCommonFiles)
         flags |= FILE_COMMON;
      fh = FileOpen(gTransactionCsvFile, flags, ',');
      if(fh == INVALID_HANDLE)
        {
         Print("Cannot open transaction CSV: ", gTransactionCsvFile, " err=", GetLastError());
         return false;
        }
     }

   if(FileSize(fh) == 0)
     {
      FileWrite(fh,
                "time",
                "deal_ticket",
                "order_ticket",
                "position_id",
                "symbol",
                "deal_type",
                "entry",
                "volume",
                "price",
                "sl",
                "tp",
                "profit",
                "swap",
                "commission",
                "comment",
                "news_summary",
                "news_signal",
                "news_name",
                "news_country",
                "news_actual",
                "news_forecast",
                "news_previous",
                "news_impact",
                "news_journal");
     }

   FileSeek(fh, 0, SEEK_END);
   string comment = HistoryDealGetString(deal_ticket, DEAL_COMMENT);
   StringReplace(comment, ",", ";");
   string news_summary = gCurrentNewsSummary;
   string news_journal = gCurrentNewsJournal;
   string news_name = gCurrentNewsSelectedName;
   string news_country = gCurrentNewsSelectedCountry;
   StringReplace(news_summary, ",", ";");
   StringReplace(news_journal, ",", ";");
   StringReplace(news_name, ",", ";");
   StringReplace(news_country, ",", ";");

   FileWrite(fh,
             TimeToString((datetime)HistoryDealGetInteger(deal_ticket, DEAL_TIME), TIME_DATE | TIME_SECONDS),
             (string)deal_ticket,
             (string)HistoryDealGetInteger(deal_ticket, DEAL_ORDER),
             (string)HistoryDealGetInteger(deal_ticket, DEAL_POSITION_ID),
             symbol,
             DealTypeToText((ENUM_DEAL_TYPE)HistoryDealGetInteger(deal_ticket, DEAL_TYPE)),
             DealEntryToText((ENUM_DEAL_ENTRY)HistoryDealGetInteger(deal_ticket, DEAL_ENTRY)),
             DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_VOLUME), 2),
             DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_PRICE), (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)),
             DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_SL), (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)),
             DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_TP), (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)),
             DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_PROFIT), 2),
             DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_SWAP), 2),
             DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_COMMISSION), 2),
             comment,
             news_summary,
             (string)gCurrentNewsSelectedSignal,
             news_name,
             news_country,
             DoubleToString(gCurrentNewsSelectedActual, 4),
             DoubleToString(gCurrentNewsSelectedForecast, 4),
             DoubleToString(gCurrentNewsSelectedPrevious, 4),
             (string)gCurrentNewsSelectedImpact,
             news_journal);
   FileClose(fh);
   return true;
  }

//+------------------------------------------------------------------+
bool ExportAllDealHistoryCsv()
  {
   if(!HistorySelect(0, TimeCurrent()))
      return false;

   int flags = FILE_WRITE | FILE_CSV | FILE_ANSI;
   if(InpUseCommonFiles)
      flags |= FILE_COMMON;

   int fh = FileOpen(gTransactionCsvFile, flags, ',');
   if(fh == INVALID_HANDLE)
     {
      Print("Cannot rewrite transaction CSV: ", gTransactionCsvFile, " err=", GetLastError());
      return false;
     }

   FileWrite(fh,
             "time",
             "deal_ticket",
             "order_ticket",
             "position_id",
             "symbol",
             "deal_type",
             "entry",
             "volume",
             "price",
             "sl",
             "tp",
             "profit",
             "swap",
             "commission",
             "comment",
             "news_summary",
             "news_signal",
             "news_name",
             "news_country",
             "news_actual",
             "news_forecast",
             "news_previous",
             "news_impact",
             "news_journal");

   int total = HistoryDealsTotal();
   for(int i = 0; i < total; i++)
     {
      ulong deal_ticket = HistoryDealGetTicket(i);
      if(deal_ticket == 0 || !HistoryDealSelect(deal_ticket))
         continue;

      string symbol = HistoryDealGetString(deal_ticket, DEAL_SYMBOL);
      if(symbol != _Symbol)
         continue;
      if((ulong)HistoryDealGetInteger(deal_ticket, DEAL_MAGIC) != InpMagicNumber)
         continue;

      string comment = HistoryDealGetString(deal_ticket, DEAL_COMMENT);
      StringReplace(comment, ",", ";");
      string news_summary = gCurrentNewsSummary;
      string news_journal = gCurrentNewsJournal;
      string news_name = gCurrentNewsSelectedName;
      string news_country = gCurrentNewsSelectedCountry;
      StringReplace(news_summary, ",", ";");
      StringReplace(news_journal, ",", ";");
      StringReplace(news_name, ",", ";");
      StringReplace(news_country, ",", ";");

      FileWrite(fh,
                TimeToString((datetime)HistoryDealGetInteger(deal_ticket, DEAL_TIME), TIME_DATE | TIME_SECONDS),
                (string)deal_ticket,
                (string)HistoryDealGetInteger(deal_ticket, DEAL_ORDER),
                (string)HistoryDealGetInteger(deal_ticket, DEAL_POSITION_ID),
                symbol,
                DealTypeToText((ENUM_DEAL_TYPE)HistoryDealGetInteger(deal_ticket, DEAL_TYPE)),
                DealEntryToText((ENUM_DEAL_ENTRY)HistoryDealGetInteger(deal_ticket, DEAL_ENTRY)),
                DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_VOLUME), 2),
                DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_PRICE), (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)),
                DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_SL), (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)),
                DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_TP), (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)),
                DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_PROFIT), 2),
                DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_SWAP), 2),
                DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_COMMISSION), 2),
                comment,
                news_summary,
                (string)gCurrentNewsSelectedSignal,
                news_name,
                news_country,
                DoubleToString(gCurrentNewsSelectedActual, 4),
                DoubleToString(gCurrentNewsSelectedForecast, 4),
                DoubleToString(gCurrentNewsSelectedPrevious, 4),
                (string)gCurrentNewsSelectedImpact,
                news_journal);
     }

   FileClose(fh);
   return true;
  }

//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result)
  {
   if(!InpEnableTransactionCsv)
      return;
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD || trans.deal == 0)
      return;

   AppendDealCsv(trans.deal);
  }
