//+------------------------------------------------------------------+
//|                                            MacroNewsOnnxTrader.mq5|
//|  News-driven ONNX EA for MT5 backtesting and replay.             |
//|  Reads macro runtime events exported by the training pipeline and |
//|  trades only when a relevant news event is released.             |
//+------------------------------------------------------------------+
#property strict
#property version   "1.00"

#include <Trade/Trade.mqh>
#include <ExpertHistory.mqh>

enum ENUM_SIGNAL_SOURCE
  {
   SIGNAL_LR = 0,
   SIGNAL_SVM = 1,
   SIGNAL_RF = 2,
   SIGNAL_XGB = 3,
   SIGNAL_ENSEMBLE = 4
  };

input string InpModelFolder = "Assets";
input ENUM_SIGNAL_SOURCE InpSignalSource = SIGNAL_ENSEMBLE;
input double InpLots = 0.10;
input int InpSlPoints = 300;
input int InpTpPoints = 600;
input bool InpUseCommonFiles = true;
input bool InpUseCpuOnly = true;
input bool InpCloseOnHold = false;
input ulong InpMagicNumber = 26072401;
input int InpCooldownBars = 12;
input double InpMinSignalEdge = 0.02;
input bool InpEnableTransactionCsv = true;
input string InpTransactionCsvFile = "";
input bool InpExportHistoryOnDeinit = true;

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
double gThreshold = 0.60;
int gClassCount = 3;

datetime gRuntimeEventTimes[];
string gRuntimeEventNames[];
double gRuntimeEventFeatures[];
int gRuntimeEventCount = 0;
int gRuntimeEventCapacity = 0;
int gNextEventIndex = 0;

datetime gLastBarTime = 0;
string gTransactionCsvFile = "";
string gResolvedConfigFile = "";
string gResolvedSignalSource = "";
string gResolvedRuntimeFile = "";
bool gRuntimeLoaded = false;
int gBarsToWaitBeforeEntry = 0;

bool LoadConfig();
bool LoadRuntimeEvents();
bool LoadModels();
void ReleaseModels();
bool RunModel(const int model_index, const vectorf &features, double &prob_down, double &prob_hold, double &prob_up);
int SignalFromProbabilities(const double prob_down, const double prob_hold, const double prob_up, const double threshold, const double signal_edge);
bool EvaluateEvent(const int event_index, double &bull_prob, double &bear_prob, int &signal);
bool ProcessNewsWindow(const datetime previous_bar_time, const datetime current_bar_time);
bool ProcessSignal(const int signal, const string trade_comment);
bool HasOpenPosition(const long position_type);
bool CloseOpenPositions();
string BuildModelPath(const string file_name);
void ApplyCooldown();
double Clamp01(const double value);
string ResolveTransactionCsvFile();
string ResolveConfigFileName();
string ResolveRuntimeFileName();
string ResolveNewsEventFileName();
string ResolveSignalSourceName();
bool SaveHistory();
void EnsureRuntimeCapacity(const int needed_count);
void SaveEventFeature(const int event_index, const int feature_index, const double value);
double GetEventFeature(const int event_index, const int feature_index);

//+------------------------------------------------------------------+
int OnInit()
  {
   ArrayResize(gOnnxHandles, gModelCount);
   ArrayResize(gModelWeights, gModelCount);
   ArrayResize(gModelFiles, gModelCount);
   ArrayResize(gModelThresholds, gModelCount);
   ArrayInitialize(gOnnxHandles, INVALID_HANDLE);
   ArrayInitialize(gModelWeights, 0.0);
   ArrayInitialize(gModelThresholds, 0.0);

   trade.SetExpertMagicNumber(InpMagicNumber);

   gResolvedConfigFile = ResolveConfigFileName();
   gResolvedRuntimeFile = ResolveRuntimeFileName();
   if(StringLen(gResolvedConfigFile) == 0 || StringLen(gResolvedRuntimeFile) == 0)
     {
      Print("Unsupported chart symbol for macro-news config mapping: ", _Symbol);
      return INIT_FAILED;
     }

   gResolvedSignalSource = ResolveSignalSourceName();
   gTransactionCsvFile = ResolveTransactionCsvFile();

   if(!LoadConfig())
      return INIT_FAILED;
   if(!LoadRuntimeEvents())
      return INIT_FAILED;
   if(!LoadModels())
      return INIT_FAILED;

   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(InpEnableTransactionCsv && InpExportHistoryOnDeinit)
      SaveHistory();
   ReleaseModels();
  }

//+------------------------------------------------------------------+
void OnTick()
  {
   MqlRates rates[];
   ArrayResize(rates, 2);
   ArraySetAsSeries(rates, true);
   if(CopyRates(_Symbol, PERIOD_H1, 0, 2, rates) < 2)
      return;

   if(rates[0].time == gLastBarTime)
      return;

   datetime current_bar_time = rates[0].time;
   datetime previous_bar_time = rates[1].time;
   if(previous_bar_time >= current_bar_time)
     {
      PrintFormat("Invalid H1 bar order for %s prev=%s current=%s",
                  _Symbol,
                  TimeToString(previous_bar_time, TIME_DATE | TIME_MINUTES),
                  TimeToString(current_bar_time, TIME_DATE | TIME_MINUTES));
      return;
     }

   gLastBarTime = current_bar_time;
   ApplyCooldown();

   if(!gRuntimeLoaded || gRuntimeEventCount <= 0)
      return;

   ProcessNewsWindow(previous_bar_time, current_bar_time);
  }

//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,
                         const MqlTradeRequest &request,
                         const MqlTradeResult &result)
  {
   return;
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
bool LoadRuntimeEvents()
  {
   int flags = FILE_READ | FILE_CSV | FILE_ANSI;
   if(InpUseCommonFiles)
      flags |= FILE_COMMON;

   string runtime_path = InpModelFolder + "\\" + gResolvedRuntimeFile;
   int handle = FileOpen(runtime_path, flags, ',');
   if(handle == INVALID_HANDLE)
     {
      Print("Failed to open runtime event file: ", runtime_path, " err=", GetLastError());
      return false;
     }

   if(FileIsEnding(handle))
     {
      FileClose(handle);
      Print("Runtime event file is empty: ", runtime_path);
      return false;
     }

   // Skip header row: Time + feature columns.
   FileReadString(handle);
   for(int i = 0; i < gFeatureCount; i++)
      FileReadString(handle);

   gRuntimeEventCount = 0;
   gRuntimeEventCapacity = 0;

   while(!FileIsEnding(handle))
     {
      string time_str = FileReadString(handle);
      if(StringLen(time_str) == 0 && FileIsEnding(handle))
         break;

      datetime event_time = StringToTime(time_str);
      if(event_time <= 0)
         continue;

      EnsureRuntimeCapacity(gRuntimeEventCount + 1);
      gRuntimeEventTimes[gRuntimeEventCount] = event_time;

      for(int i = 0; i < gFeatureCount; i++)
        {
         string raw_str = FileReadString(handle);
         double raw_value = StringToDouble(raw_str);
         double std = gFeatureStds[i];
         if(std == 0.0)
            std = 1.0;
         SaveEventFeature(gRuntimeEventCount, i, (raw_value - gFeatureMeans[i]) / std);
        }

      gRuntimeEventCount++;
     }

   FileClose(handle);

   ArrayResize(gRuntimeEventNames, 0);

   string news_path = InpModelFolder + "\\" + ResolveNewsEventFileName();
   int news_handle = FileOpen(news_path, flags, ',');
   if(news_handle != INVALID_HANDLE)
     {
      if(!FileIsEnding(news_handle))
        {
         while(!FileIsLineEnding(news_handle) && !FileIsEnding(news_handle))
            FileReadString(news_handle);

         int news_index = 0;
         while(!FileIsEnding(news_handle) && news_index < gRuntimeEventCount)
           {
            string event_time_str = FileReadString(news_handle);
            if(StringLen(event_time_str) == 0 && FileIsEnding(news_handle))
               break;

            string event_name = FileReadString(news_handle);
            if(news_index >= ArraySize(gRuntimeEventNames))
               ArrayResize(gRuntimeEventNames, news_index + 1);
            gRuntimeEventNames[news_index] = event_name;

            while(!FileIsLineEnding(news_handle) && !FileIsEnding(news_handle))
               FileReadString(news_handle);
            news_index++;
           }
        }
      FileClose(news_handle);
     }
   else
     {
      Print("Failed to open news event detail file: ", news_path, " err=", GetLastError());
     }

   if(ArraySize(gRuntimeEventNames) < gRuntimeEventCount)
      ArrayResize(gRuntimeEventNames, gRuntimeEventCount);

   gRuntimeLoaded = (gRuntimeEventCount > 0);
   if(!gRuntimeLoaded)
      Print("No runtime events loaded after parsing: ", runtime_path);
   else
     {
      datetime first_event = gRuntimeEventTimes[0];
      datetime last_event = gRuntimeEventTimes[gRuntimeEventCount - 1];
      Print("Loaded ", gRuntimeEventCount, " news events from ", runtime_path,
            " range=", TimeToString(first_event, TIME_DATE | TIME_MINUTES),
            " .. ", TimeToString(last_event, TIME_DATE | TIME_MINUTES));
     }
   return gRuntimeLoaded;
  }

//+------------------------------------------------------------------+
bool LoadModels()
  {
   uint onnx_flags = 0;
   if(InpUseCommonFiles)
      onnx_flags |= ONNX_COMMON_FOLDER;
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
int SignalFromProbabilities(const double prob_down, const double prob_hold, const double prob_up, const double threshold, const double signal_edge)
  {
   if(prob_up >= threshold && prob_up >= prob_hold && prob_up >= prob_down && (prob_up - MathMax(prob_hold, prob_down)) >= signal_edge)
      return 1;
   if(prob_down >= threshold && prob_down >= prob_hold && prob_down >= prob_up && (prob_down - MathMax(prob_hold, prob_up)) >= signal_edge)
      return -1;
   return 0;
  }

//+------------------------------------------------------------------+
bool EvaluateEvent(const int event_index, double &bull_prob, double &bear_prob, int &signal)
  {
   vectorf features(gFeatureCount);
   for(int i = 0; i < gFeatureCount; i++)
      features[i] = (float)GetEventFeature(event_index, i);

   double down = 0.0;
   double hold = 0.0;
   double up = 0.0;

   if(InpSignalSource == SIGNAL_ENSEMBLE)
     {
      double weight_total = 0.0;
      for(int i = 0; i < gModelCount; i++)
         weight_total += MathMax(gModelWeights[i], 0.0);
      if(weight_total <= 0.0)
         weight_total = (double)gModelCount;

      for(int i = 0; i < gModelCount; i++)
        {
         double model_down = 0.0, model_hold = 0.0, model_up = 0.0;
         if(!RunModel(i, features, model_down, model_hold, model_up))
            return false;
         double weight = MathMax(gModelWeights[i], 0.0);
         down += weight * model_down;
         hold += weight * model_hold;
         up += weight * model_up;
        }

      down /= weight_total;
      hold /= weight_total;
      up /= weight_total;
      bull_prob = up;
      bear_prob = down;
      double effective_threshold = gThreshold;
      if(effective_threshold > 0.45)
         effective_threshold = 0.45;
      signal = SignalFromProbabilities(down, hold, up, effective_threshold, InpMinSignalEdge);
      return true;
     }

   int model_index = (int)InpSignalSource;
   if(model_index < 0 || model_index >= gModelCount)
      model_index = 3;

   if(!RunModel(model_index, features, down, hold, up))
      return false;

   double threshold = gModelThresholds[model_index];
   if(threshold <= 0.0)
      threshold = gThreshold;
   if(threshold > 0.45)
      threshold = 0.45;
   bull_prob = up;
   bear_prob = down;
   signal = SignalFromProbabilities(down, hold, up, threshold, InpMinSignalEdge);
   return true;
  }

//+------------------------------------------------------------------+
bool ProcessNewsWindow(const datetime previous_bar_time, const datetime current_bar_time)
  {
   while(gNextEventIndex < gRuntimeEventCount && gRuntimeEventTimes[gNextEventIndex] < previous_bar_time)
      gNextEventIndex++;

   int idx = gNextEventIndex;
   int event_count = 0;
   int buy_votes = 0;
   int sell_votes = 0;
   double bull_sum = 0.0;
   double bear_sum = 0.0;
   double best_bull_strength = -1.0;
   double best_bear_strength = -1.0;
   string best_bull_event_name = "";
   string best_bear_event_name = "";

   while(idx < gRuntimeEventCount && gRuntimeEventTimes[idx] < current_bar_time)
     {
      if(gRuntimeEventTimes[idx] >= previous_bar_time)
        {
         double bull_prob = 0.0;
         double bear_prob = 0.0;
         int signal = 0;
         if(EvaluateEvent(idx, bull_prob, bear_prob, signal))
           {
            bull_sum += bull_prob;
            bear_sum += bear_prob;
            if(signal == 1)
               buy_votes++;
            else if(signal == -1)
               sell_votes++;
            string event_name = "";
            if(idx < ArraySize(gRuntimeEventNames))
               event_name = gRuntimeEventNames[idx];
            if(StringLen(event_name) == 0)
               event_name = "Unnamed News Event";

            // Keep the strongest bullish and bearish event separately. The final
            // trade comment is chosen after the window-level signal is known.
            if(bull_prob > best_bull_strength)
              {
               best_bull_strength = bull_prob;
               best_bull_event_name = event_name;
              }
            if(bear_prob > best_bear_strength)
              {
               best_bear_strength = bear_prob;
               best_bear_event_name = event_name;
              }
            event_count++;
           }
        }
      idx++;
     }

   gNextEventIndex = idx;
   if(event_count <= 0)
      return true;

   double bull_avg = bull_sum / event_count;
   double bear_avg = bear_sum / event_count;
   int signal = 0;

   if(buy_votes > sell_votes)
      signal = 1;
   else if(sell_votes > buy_votes)
      signal = -1;
   else if(buy_votes == sell_votes)
     {
      if((bull_avg - bear_avg) >= InpMinSignalEdge)
         signal = 1;
      else if((bear_avg - bull_avg) >= InpMinSignalEdge)
         signal = -1;
     }

   string signal_text = "WAIT";
   if(signal == 1)
      signal_text = "BUY";
   else if(signal == -1)
      signal_text = "SELL";

   string selected_event_name = "";
   if(signal == 1)
      selected_event_name = best_bull_event_name;
   else if(signal == -1)
      selected_event_name = best_bear_event_name;

   PrintFormat("%s %s window=%s..%s events=%d votes(B/S)=%d/%d signal=%s bull=%.4f bear=%.4f",
               _Symbol,
               gResolvedSignalSource,
               TimeToString(previous_bar_time, TIME_DATE | TIME_MINUTES),
               TimeToString(current_bar_time, TIME_DATE | TIME_MINUTES),
               event_count,
               buy_votes,
               sell_votes,
               signal_text,
               bull_avg,
               bear_avg);
   if(signal != 0)
      PrintFormat("%s %s selected_event=%s", _Symbol, gResolvedSignalSource, selected_event_name);

   return ProcessSignal(signal, selected_event_name);
  }

//+------------------------------------------------------------------+
bool ProcessSignal(const int signal, const string trade_comment)
  {
   bool has_buy = HasOpenPosition(POSITION_TYPE_BUY);
   bool has_sell = HasOpenPosition(POSITION_TYPE_SELL);
   string order_comment = trade_comment;
   if(StringLen(order_comment) == 0)
      order_comment = "Unnamed News Event";
   PrintFormat("%s %s order_comment=%s", _Symbol, gResolvedSignalSource, order_comment);

   if(signal == 1)
     {
      if(gBarsToWaitBeforeEntry > 0)
         return true;
      if(has_sell && !CloseOpenPositions())
         return false;
      if(!has_buy)
        {
         double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         double sl = (InpSlPoints > 0) ? ask - InpSlPoints * _Point : 0.0;
         double tp = (InpTpPoints > 0) ? ask + InpTpPoints * _Point : 0.0;
         bool ok = trade.Buy(InpLots, _Symbol, ask, sl, tp, order_comment);
         if(ok)
            gBarsToWaitBeforeEntry = MathMax(InpCooldownBars, 0);
         return ok;
        }
      return true;
     }

   if(signal == -1)
     {
      if(gBarsToWaitBeforeEntry > 0)
         return true;
      if(has_buy && !CloseOpenPositions())
         return false;
      if(!has_sell)
        {
         double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         double sl = (InpSlPoints > 0) ? bid + InpSlPoints * _Point : 0.0;
         double tp = (InpTpPoints > 0) ? bid - InpTpPoints * _Point : 0.0;
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

   return "MacroNewsOnnxTrader_" + gResolvedSignalSource + "_" + _Symbol + ".csv";
  }

//+------------------------------------------------------------------+
string ResolveConfigFileName()
  {
   string symbol = _Symbol;
   StringToUpper(symbol);

   if(symbol == "EURUSD")
      return "macro_config_eurusd.csv";
   if(symbol == "GBPUSD")
      return "macro_config_gbpusd.csv";
   if(symbol == "AUDUSD")
      return "macro_config_audusd.csv";
   if(symbol == "EURJPY")
      return "macro_config_eurjpy.csv";
   if(symbol == "USDJPY")
      return "macro_config_usdjpy.csv";

   return "";
  }

//+------------------------------------------------------------------+
string ResolveRuntimeFileName()
  {
   string symbol = _Symbol;
   StringToUpper(symbol);

   if(symbol == "EURUSD")
      return "macro_runtime_events_eurusd.csv";
   if(symbol == "GBPUSD")
      return "macro_runtime_events_gbpusd.csv";
   if(symbol == "AUDUSD")
      return "macro_runtime_events_audusd.csv";
   if(symbol == "EURJPY")
      return "macro_runtime_events_eurjpy.csv";
   if(symbol == "USDJPY")
      return "macro_runtime_events_usdjpy.csv";

  return "";
  }

//+------------------------------------------------------------------+
string ResolveNewsEventFileName()
  {
   string symbol = _Symbol;
   StringToUpper(symbol);

   if(symbol == "EURUSD")
      return "macro_news_events_eurusd.csv";
   if(symbol == "GBPUSD")
      return "macro_news_events_gbpusd.csv";
   if(symbol == "AUDUSD")
      return "macro_news_events_audusd.csv";
   if(symbol == "EURJPY")
      return "macro_news_events_eurjpy.csv";
   if(symbol == "USDJPY")
      return "macro_news_events_usdjpy.csv";

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
bool SaveHistory()
  {
   if(!InpEnableTransactionCsv)
      return false;

   string separator = ",";
   string decimal_point = ".";
   uint common_flag = 0;
   if(InpUseCommonFiles)
      common_flag = FILE_COMMON;

   CExpertHistory account_history("MacroNewsOnnxTrader", "", separator, decimal_point);
   account_history.Export(gTransactionCsvFile, HEF_CSV_DEALS, HFF_ACCOUNT_PERIOD, common_flag);
   Print("Trade history exported to: ", gTransactionCsvFile);
   return true;
  }

//+------------------------------------------------------------------+
void EnsureRuntimeCapacity(const int needed_count)
  {
   if(needed_count <= gRuntimeEventCapacity)
      return;

   int new_capacity = gRuntimeEventCapacity;
   if(new_capacity <= 0)
      new_capacity = 512;
   while(new_capacity < needed_count)
      new_capacity *= 2;

   ArrayResize(gRuntimeEventTimes, new_capacity);
   ArrayResize(gRuntimeEventFeatures, new_capacity * gFeatureCount);
   gRuntimeEventCapacity = new_capacity;
  }

//+------------------------------------------------------------------+
void SaveEventFeature(const int event_index, const int feature_index, const double value)
  {
   gRuntimeEventFeatures[(event_index * gFeatureCount) + feature_index] = value;
  }

//+------------------------------------------------------------------+
double GetEventFeature(const int event_index, const int feature_index)
  {
   return gRuntimeEventFeatures[(event_index * gFeatureCount) + feature_index];
  }

//+------------------------------------------------------------------+
