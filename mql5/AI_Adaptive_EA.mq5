//+------------------------------------------------------------------+
//|                                             AI_Adaptive_EA.mq5    |
//|  EA d'exécution pour le moteur IA de régime/impulsion.            |
//|                                                                    |
//|  Architecture : Python (regime_engine) = intelligence -> API HTTP  |
//|  Cet EA = SEULE couche d'exécution, avec son PROPRE moteur         |
//|  de risque en dur. Même si l'API répond n'importe quoi,            |
//|  l'EA ne dépasse JAMAIS les limites ci-dessous.                    |
//|                                                                    |
//|  Prérequis MT5 : Outils -> Options -> Expert Advisors ->            |
//|  autoriser WebRequest pour l'URL de ton API Render.                 |
//+------------------------------------------------------------------+
#property strict

input string  InpApiUrl              = "https://ai-adaptive-trading-engine.onrender.com/signal";
input int     InpCandlesToSend       = 200;
input int     InpTimerSeconds        = 60;

input double  InpMaxRiskPerTradePct  = 0.5;
input double  InpMaxDailyLossPct     = 2.0;
input double  InpMaxDrawdownPct      = 8.0;
input int     InpMaxOpenPositions    = 3;
input double  InpMaxSpreadPoints     = 300;
input int     InpMaxConsecutiveLoss  = 4;
input double  InpMinImpulseProba     = 0.65;
input int     InpMaxTradesPerHour    = 6;
input int     InpAtrPeriodForSL      = 14;
input double  InpAtrMultipleForSL    = 1.5;

double   g_dayStartEquity   = 0.0;
double   g_equityPeak       = 0.0;
int      g_consecutiveLosses= 0;
int      g_tradesThisHour   = 0;
datetime g_currentHourStart = 0;
datetime g_currentDay       = 0;
bool     g_halted           = false;
string   g_haltReason       = "";

int      g_atrHandle;

int OnInit()
  {
   g_dayStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   g_equityPeak      = AccountInfoDouble(ACCOUNT_EQUITY);
   g_currentDay       = TimeCurrent();
   g_currentHourStart = TimeCurrent();
   g_atrHandle = iATR(_Symbol, PERIOD_CURRENT, InpAtrPeriodForSL);

   EventSetTimer(InpTimerSeconds);
   Print("AI_Adaptive_EA initialisé. Moteur de risque local ACTIF.");
   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason)
  {
   EventKillTimer();
  }

void OnTimer()
  {
   RolloverDailyAndHourlyCountersIfNeeded();
   UpdateEquityPeakAndDrawdown();

   if(g_halted)
     {
      Comment("AI_Adaptive_EA À L'ARRÊT : ", g_haltReason, "\n(Reset manuel requis après vérification.)");
      return;
     }

   string json = BuildCandlesJson();
   if(json == "")
      return;

   string response;
   if(!CallApi(json, response))
     {
      Print("Erreur d'appel API - aucune action.");
      return;
     }

   string regime, decision;
   double probaHausse, probaBaisse, probaNone;
   bool confidenceOk;
   if(!ParseSignalResponse(response, regime, probaHausse, probaBaisse, probaNone, decision, confidenceOk))
     {
      Print("Réponse API invalide : ", response);
      return;
     }

   Comment("Régime: ", regime,
           " | P(hausse)=", DoubleToString(probaHausse,2),
           " | P(baisse)=", DoubleToString(probaBaisse,2),
           " | Décision IA: ", decision);

   if(!confidenceOk || decision == "aucun_trade")
      return;

   double impulseProba = (decision == "achat") ? probaHausse : probaBaisse;
   TryExecuteTrade(decision, impulseProba);
  }

bool RiskCheckNewTrade(double impulseProba, string &reasonOut)
  {
   if(g_halted) { reasonOut = "Système à l'arrêt : " + g_haltReason; return false; }

   if(impulseProba < InpMinImpulseProba)
     { reasonOut = "Probabilité d'edge insuffisante"; return false; }

   double spreadPoints = (double)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
   if(spreadPoints > InpMaxSpreadPoints)
     { reasonOut = "Spread trop élevé"; return false; }

   int openPositions = CountOpenPositions();
   if(openPositions >= InpMaxOpenPositions)
     { reasonOut = "Nombre max de positions atteint"; return false; }

   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double dailyLossPct = 100.0 * (g_dayStartEquity - equity) / g_dayStartEquity;
   if(dailyLossPct >= InpMaxDailyLossPct)
     { Halt("Perte quotidienne max atteinte (" + DoubleToString(dailyLossPct,2) + "%)"); reasonOut = g_haltReason; return false; }

   double ddPct = 100.0 * (g_equityPeak - equity) / g_equityPeak;
   if(ddPct >= InpMaxDrawdownPct)
     { Halt("Drawdown max atteint (" + DoubleToString(ddPct,2) + "%)"); reasonOut = g_haltReason; return false; }

   if(g_consecutiveLosses >= InpMaxConsecutiveLoss)
     { Halt(IntegerToString(g_consecutiveLosses) + " pertes consécutives"); reasonOut = g_haltReason; return false; }

   if(g_tradesThisHour >= InpMaxTradesPerHour)
     { reasonOut = "Max trades/heure atteint"; return false; }

   reasonOut = "OK";
   return true;
  }

void TryExecuteTrade(string decision, double impulseProba)
  {
   string reason;
   if(!RiskCheckNewTrade(impulseProba, reason))
     {
      Print("Trade refusé par le moteur de risque : ", reason);
      return;
     }

   double atrBuf[];
   ArraySetAsSeries(atrBuf, true);
   if(CopyBuffer(g_atrHandle, 0, 0, 1, atrBuf) <= 0)
      return;
   double atr = atrBuf[0];

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);

   double entry, sl, tp;
   ENUM_ORDER_TYPE type;
   if(decision == "achat")
     {
      type = ORDER_TYPE_BUY;
      entry = ask;
      sl = entry - atr * InpAtrMultipleForSL;
      tp = entry + atr * InpAtrMultipleForSL * 1.5;
     }
   else
     {
      type = ORDER_TYPE_SELL;
      entry = bid;
      sl = entry + atr * InpAtrMultipleForSL;
      tp = entry - atr * InpAtrMultipleForSL * 1.5;
     }

   double lots = CalcLotsForRisk(entry, sl);
   if(lots <= 0)
     {
      Print("Taille de position calculée = 0, trade annulé.");
      return;
     }

   MqlTradeRequest request;
   MqlTradeResult  result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action    = TRADE_ACTION_DEAL;
   request.symbol    = _Symbol;
   request.volume    = lots;
   request.type      = type;
   request.price     = entry;
   request.sl        = NormalizeDouble(sl, _Digits);
   request.tp        = NormalizeDouble(tp, _Digits);
   request.deviation = (int)InpMaxSpreadPoints;
   request.magic     = 20260812;
   request.comment    = "AI_Adaptive_EA";

   if(OrderSend(request, result))
     {
      g_tradesThisHour++;
      Print("Trade exécuté : ", decision, " lots=", lots, " sl=", sl, " tp=", tp);
     }
   else
     {
      Print("Échec OrderSend : ", result.retcode, " ", result.comment);
     }
  }

double CalcLotsForRisk(double entry, double sl)
  {
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskAmount = balance * (InpMaxRiskPerTradePct / 100.0);
   double stopDistance = MathAbs(entry - sl);
   if(stopDistance <= 0) return 0.0;

   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(tickValue <= 0 || tickSize <= 0) return 0.0;

   double valuePerPriceUnit = tickValue / tickSize;
   double lots = riskAmount / (stopDistance * valuePerPriceUnit);

   double minLot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   lots = MathFloor(lots / lotStep) * lotStep;
   lots = MathMax(lots, minLot);
   lots = MathMin(lots, maxLot);
   return lots;
  }

int CountOpenPositions()
  {
   int count = 0;
   for(int i = 0; i < PositionsTotal(); i++)
      if(PositionGetSymbol(i) == _Symbol)
         count++;
   return count;
  }

void UpdateEquityPeakAndDrawdown()
  {
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   if(equity > g_equityPeak)
      g_equityPeak = equity;
  }

void RolloverDailyAndHourlyCountersIfNeeded()
  {
   MqlDateTime now, dayRef, hourRef;
   TimeToStruct(TimeCurrent(), now);
   TimeToStruct(g_currentDay, dayRef);
   TimeToStruct(g_currentHourStart, hourRef);

   if(now.day != dayRef.day || now.mon != dayRef.mon)
     {
      g_dayStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);
      g_currentDay = TimeCurrent();
     }

   if(now.hour != hourRef.hour)
     {
      g_tradesThisHour = 0;
      g_currentHourStart = TimeCurrent();
     }
  }

void Halt(string reason)
  {
   g_halted = true;
   g_haltReason = reason;
   Print("*** ARRÊT AUTOMATIQUE DU BOT : ", reason, " ***");
  }

void OnTradeTransaction(const MqlTradeTransaction &trans,
                         const MqlTradeRequest &request,
                         const MqlTradeResult &result)
  {
   if(trans.type == TRADE_TRANSACTION_DEAL_ADD)
     {
      if(HistoryDealSelect(trans.deal))
        {
         double profit = HistoryDealGetDouble(trans.deal, DEAL_PROFIT);
         if(profit < 0)
            g_consecutiveLosses++;
         else if(profit > 0)
            g_consecutiveLosses = 0;
        }
     }
  }

string BuildCandlesJson()
  {
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   int copied = CopyRates(_Symbol, PERIOD_CURRENT, 0, InpCandlesToSend, rates);
   if(copied <= 0)
      return "";

   string arr = "[";
   for(int i = copied - 1; i >= 0; i--)
     {
      arr += StringFormat(
         "{\"time\":\"%s\",\"open\":%.5f,\"high\":%.5f,\"low\":%.5f,\"close\":%.5f,\"tick_volume\":%d,\"spread\":%d}",
         TimeToString(rates[i].time, TIME_DATE|TIME_SECONDS),
         rates[i].open, rates[i].high, rates[i].low, rates[i].close,
         (int)rates[i].tick_volume, rates[i].spread);
      if(i > 0) arr += ",";
     }
   arr += "]";

   return StringFormat("{\"symbol\":\"%s\",\"candles\":%s}", _Symbol, arr);
  }

bool CallApi(string jsonBody, string &responseOut)
  {
   char post[], result[];
   string headers = "Content-Type: application/json\r\n";
   StringToCharArray(jsonBody, post, 0, StringLen(jsonBody));

   string resultHeaders;
   int res = WebRequest("POST", InpApiUrl, headers, 5000, post, result, resultHeaders);
   if(res == -1)
     {
      Print("WebRequest échec, erreur=", GetLastError(),
            ". Vérifie que l'URL est autorisée dans Options -> Expert Advisors.");
      return false;
     }
   responseOut = CharArrayToString(result);
   return true;
  }

bool ParseSignalResponse(string json, string &regime, double &probaHausse,
                          double &probaBaisse, double &probaNone,
                          string &decision, bool &confidenceOk)
  {
   regime      = ExtractJsonString(json, "regime");
   decision    = ExtractJsonString(json, "decision");
   probaHausse = ExtractJsonNumber(json, "proba_hausse");
   probaBaisse = ExtractJsonNumber(json, "proba_baisse");
   probaNone   = ExtractJsonNumber(json, "proba_none");
   confidenceOk = (StringFind(json, "\"confidence_ok\":true") >= 0);
   return (decision != "");
  }

string ExtractJsonString(string json, string key)
  {
   string pattern = "\"" + key + "\":\"";
   int p = StringFind(json, pattern);
   if(p < 0) return "";
   p += StringLen(pattern);
   int e = StringFind(json, "\"", p);
   if(e < 0) return "";
   return StringSubstr(json, p, e - p);
  }

double ExtractJsonNumber(string json, string key)
  {
   string pattern = "\"" + key + "\":";
   int p = StringFind(json, pattern);
   if(p < 0) return 0.0;
   p += StringLen(pattern);
   int e = p;
   while(e < StringLen(json) && (StringGetCharacter(json, e) != ',' && StringGetCharacter(json, e) != '}'))
      e++;
   return StringToDouble(StringSubstr(json, p, e - p));
  }
