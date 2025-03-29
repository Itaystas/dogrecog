package dev.lordyorden.tradely.models

import android.util.Log
import com.github.mikephil.charting.data.CandleEntry
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import java.time.LocalDate
import java.time.LocalDateTime
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter

object StockParser {

    val baseDate: LocalDateTime = LocalDateTime.of(2020, 1, 1, 0 ,0 ,0)

    private fun dateStringToDate(date: String, format: String = "yyyy-MM-dd"): LocalDate {
        val formatter = DateTimeFormatter.ofPattern(format)
        val localDate = LocalDate.parse(date, formatter)
        return localDate
    }

    private fun dateToNormalizedFloat(date: LocalDate, minDate: LocalDate = baseDate.toLocalDate()): Float {
        return (date.toEpochDay() - minDate.toEpochDay()).toFloat()
    }

    private fun dateTimeToNormalizedFloat(dateTime: LocalDateTime, minDateTime: LocalDateTime = baseDate): Float {
        val hoursSinceBase = dateTime.toEpochSecond(ZoneOffset.UTC) / 360 - minDateTime.toEpochSecond(ZoneOffset.UTC) / 3606
        return hoursSinceBase.toFloat()
    }

    private fun dateTimeStringToDateTime(currDateTime: String, format: String = "yyyy-MM-dd HH:mm:ss"): LocalDateTime {
        val formatter = DateTimeFormatter.ofPattern(format)
        val localDateTime = LocalDateTime.parse(currDateTime, formatter)
        return localDateTime
    }

    private fun parseMetaData(obj: JsonObject): String? {
        try {
            if (obj.has("Error Message")) {
                return null
            }

            if (obj.has("Meta Data")) {
                val metadata = obj.get("Meta Data").asJsonObject
                val currDate = metadata.get("3. Last Refreshed").asString
                return currDate
            }
        } catch (e: IllegalStateException) {
            e.message?.let { Log.e("MetaData parsing", it) }
        } catch (e: NullPointerException) {
            e.message?.let { Log.e("MetaData parsing", it) }
        }
        return null
    }

    private fun parseTimeSeries(obj: JsonObject, title: String, type: String, minDate: LocalDateTime = baseDate): List<CandleEntry>? {

        val entries: MutableList<CandleEntry> = mutableListOf()
        try {
            if (obj.has(title)) {
                val series = obj.get(title).asJsonObject

                series.asMap().forEach{(key, time) ->
                    val price = time.asJsonObject
                    val open = price.get("1. open").asFloat
                    val high = price.get("2. high").asFloat
                    val low = price.get("3. low").asFloat
                    val close = price.get("4. close").asFloat

                    val date: LocalDateTime
                    val dateFloat: Float

                    if (type == "hourly"){
                        date = dateTimeStringToDateTime(key)
                        dateFloat = dateTimeToNormalizedFloat(date)
                    }else{
                        date = dateStringToDate(key).atTime(0,0,0)
                        dateFloat = dateToNormalizedFloat(date.toLocalDate())
                    }

                    if(date > minDate){
                        entries.add(CandleEntry(dateFloat, high, low, open , close))
                    }
                }
                return entries.reversed()
            }
        } catch (e: IllegalStateException) {
            e.message?.let { Log.e("MetaData parsing", it) }
        } catch (e: NullPointerException) {
            e.message?.let { Log.e("MetaData parsing", it) }
        }
        return null
    }

    fun parseWeeklyData(toParse: String): List<CandleEntry>? {
        try {
            val weekly = JsonParser.parseString(toParse).asJsonObject
            val currDate = parseMetaData(weekly)
            currDate?.let {
                val maxDate = dateStringToDate(currDate)
                val minDate = maxDate.minusYears(1)
                val entries = parseTimeSeries(weekly, "Weekly Time Series","weekly", minDate.atTime(0,0,0,))
                return entries
            }

        } catch (e: IllegalStateException) {
            e.message?.let { Log.e("weekly data parsing", it) }
        }
        return null
    }

    fun parseDailyData(toParse: String): List<CandleEntry>? {
        try {
            val daily = JsonParser.parseString(toParse).asJsonObject
            val currDate = parseMetaData(daily)

            currDate?.let {
                val maxDate = dateStringToDate(currDate)
                val minDate = maxDate.minusMonths(1)
                val entries = parseTimeSeries(daily, "Time Series (Daily)", "daily", minDate.atTime(0,0,0,))
                return entries
            }

        } catch (e: IllegalStateException) {
            e.message?.let { Log.e("daily data parsing", it) }
        }
        return null
    }

    fun parseMonthlyData(toParse: String): List<CandleEntry>? {
        try {
            val monthly = JsonParser.parseString(toParse).asJsonObject
            val currDate = parseMetaData(monthly)

            currDate?.let {
                val maxDate = dateStringToDate(currDate)
                val minDate = maxDate.minusYears(5)
                val entries = parseTimeSeries(monthly, "Monthly Time Series", "monthly", minDate.atTime(0,0,0,))
                return entries
            }

        } catch (e: IllegalStateException) {
            e.message?.let { Log.e("monthly data parsing", it) }
        }
        return null
    }

    fun parseHourlyData(toParse: String): List<CandleEntry>?{
        try {
            val hourly = JsonParser.parseString(toParse).asJsonObject
            val currDateTime = parseMetaData(hourly)

            currDateTime?.let {
                val maxDateTime = dateTimeStringToDateTime(currDateTime)
                val minDateTime = maxDateTime.toLocalDate().atTime(0,0,0,)
                val entries = parseTimeSeries(hourly, "Time Series (30min)", "hourly", minDateTime)
                return entries
            }

        } catch (e: IllegalStateException) {
            e.message?.let { Log.e("hourly data parsing", it) }
        }
        return null
    }
}