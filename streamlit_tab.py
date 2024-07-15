import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import os
from llama_index.core import PromptTemplate
from llama_index.llms.groq import Groq
from dotenv import load_dotenv
from streamlit_lottie import st_lottie
import requests
from pandasai import SmartDataframe
from pandasai.connectors.yahoo_finance import YahooFinanceConnector
from langchain_groq import ChatGroq
import logging
from duckduckgo_search import DDGS
from typing import List, Dict, Optional


# Load environment variables
load_dotenv()

# Load Groq API key from environment
GROOQ_API_KEY = os.getenv("GROOQ_API_KEY")

# Initialize Groq LLM instance
llm = Groq(model="llama3-70b-8192", api_key=GROOQ_API_KEY)
langchain_groq = ChatGroq(model="llama3-70b-8192", api_key=GROOQ_API_KEY, temperature=0.7,)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Function to analyze stock data
def analyze_stock_data(technical_data, fundamental_data, recommendations):
    template = (
        "Anda bertindak sebagai Analis saham yang bertugas menganalisa data saham sesuai dengan data yang disajikan. "
        "Tuliskan hasil analisa anda dalam bahasa indonesia yang sederhana. pastikan anda menggunakan mata uang yang sesuai dengan negara yang asal saham tersebut. Anda bisa melihat dari informasi Currency\n\n"
        "Analisis data saham berikut menggunakan analisis teknikal dan fundamental:\n"
        "Data Teknikal (Harga Buka dan Tutup):\n{technical_data}\n\n"
        "Data Fundamental:\n{fundamental_data}\n\n"
        "Berikan wawasan tentang:\n"
        "1. Tren keseluruhan, tuliskan dengan menyertakan data, angka, Jelaskan dengan narasi yang informatif\n"
        "2. Indikator teknikal utama,tuliskan dengan menyertakan data, angka, Jelaskan dengan narasi yang informatif\n"
        "3. Metrik fundamental penting,tuliskan dengan menyertakan data, angka, Jelaskan dengan narasi yang informatif\n"
        "4. Kekuatan dan kelemahan potensial,tuliskan dengan menyertakan data, angka, Jelaskan dengan narasi yang informatif\n"
        "5. Pola atau anomali yang dapat diperhatikan,tuliskan dengan menyertakan data, angka, Jelaskan dengan narasi yang informatif\n"
        "6. Rekomendasi posisi beli dan jual contoh format penulisan\n"
        "Rekomendasi Beli: Saham ini dapat dibeli jika harga saham turun ke level support sekitar 9.500, dengan target harga sekitar 10.500.\n"
        "Rekomendasi Jual: Saham ini dapat dijual jika harga saham mencapai level resistance sekitar 10.500, dengan target harga sekitar 9.500."
        "7. Rekomendasi hold\n"
        "Rekomendasi ini memadukan data dari {technical_data}{fundamental_data} dan {recommendations} buatkan narasi yang mejelaskan rekomendasi ini dan buat dalam format table\n"
    )
    prompt = PromptTemplate(template)

    formatted_prompt = prompt.format(technical_data=technical_data, fundamental_data=fundamental_data, recommendations=recommendations)
    
    response = llm.complete(formatted_prompt)
    return response

def analyze_extra_prompt(extra_prompt, analysis):
    template = (
        "Tulis Anda tidak bertanya apabila {extra_prompt} kosong.Sebagai seorang analis saham, berdasarkan data dari {analysis}, jawab pertanyaan yang diajukan oleh user ini: {extra_prompt}. Jawaban harus informatif dan tidak bertele-tele. Apabila Anda tidak memiliki jawabannya, bisa Anda tuliskan Saya tidak memiliki data yang cukup.",
    )
    prompt = PromptTemplate(template)
    formatted_prompt = prompt.format(analysis=analysis, extra_prompt=extra_prompt)
    extra_prompt_response = llm.complete(formatted_prompt)
    
    return extra_prompt_response

def extract_specific_fundamental_data(details):
    specific_data = {
        "Current Price (Harga Saham Saat Ini)": details.get("currentPrice", "N/A"),
        "Market Cap (Kapitalisasi Pasar)": details.get("marketCap", "N/A"),
        "Price to Earnings Ratio (P/E Ratio)": {
            "Trailing P/E": details.get("trailingPE", "N/A"),
            "Forward P/E": details.get("forwardPE", "N/A")
        },
        "Dividend Yield (Hasil Dividen)": details.get("dividendYield", "N/A"),
        "Return on Equity (ROE)": details.get("returnOnEquity", "N/A")
    }
    return specific_data

def format_specific_fundamental_data(specific_data):
    formatted_data = (
        f"**Current Price (Harga Saham Saat Ini)**: {specific_data['Current Price (Harga Saham Saat Ini)']} IDR\n\n"
        f"**Market Cap (Kapitalisasi Pasar)**: {specific_data['Market Cap (Kapitalisasi Pasar)']} IDR\n\n"
        f"**Price to Earnings Ratio (P/E Ratio)**:\n"
        f"  - Trailing P/E: {specific_data['Price to Earnings Ratio (P/E Ratio)']['Trailing P/E']}\n"
        f"  - Forward P/E: {specific_data['Price to Earnings Ratio (P/E Ratio)']['Forward P/E']}\n\n"
        f"**Dividend Yield (Hasil Dividen)**: {specific_data['Dividend Yield (Hasil Dividen)']}%\n\n"
        f"**Return on Equity (ROE)**: {specific_data['Return on Equity (ROE)']}%"
    )
    return formatted_data

def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def format_recommendations(recommendations_df):
    formatted_str = ""
    for index, row in recommendations_df.iterrows():
        period = row['period']
        strong_buy = row['strongBuy']
        buy = row['buy']
        hold = row['hold']
        sell = row['sell']
        strong_sell = row['strongSell']
        
        formatted_str += f"Period: {period}\n"
        formatted_str += f"Strong Buy: {strong_buy}\n"
        formatted_str += f"Buy: {buy}\n"
        formatted_str += f"Hold: {hold}\n"
        formatted_str += f"Sell: {sell}\n"
        formatted_str += f"Strong Sell: {strong_sell}\n\n"
    
    return formatted_str


class DisplayHandler:
    def __init__(self, response):
        self.response = response    
        self._validate_response()

    def _validate_response(self):
        if isinstance(self.response, str):
            self.response = {'text': self.response}

    def display(self):
        if 'image' in self.response:
            self.display_image(self.response['image'])
        elif 'chart' in self.response:
            self.display_chart(self.response['chart'])
        elif 'line_chart' in self.response:
            self.display_line_chart(self.response['line_chart'])
        else:
            st.write(self.response['text'])

    def display_image(self, image_url):
        st.image(image_url, use_column_width=True)

    def display_chart(self, chart_data):
        fig = go.Figure(data=[go.Bar(x=chart_data['x'], y=chart_data['y'])])
        fig.update_layout(title=chart_data['title'], xaxis_title=chart_data['xaxis_title'], yaxis_title=chart_data['yaxis_title'])
        st.plotly_chart(fig)

    def display_line_chart(self, line_chart_data):
        fig = go.Figure()
        for trace in line_chart_data['traces']:
            fig.add_trace(go.Scatter(x=trace['x'], y=trace['y'], mode=trace['mode'], name=trace['name']))
        fig.update_layout(title=line_chart_data['title'], xaxis_title=line_chart_data['xaxis_title'], yaxis_title=line_chart_data['yaxis_title'])
        st.plotly_chart(fig)


# Streamlit Interface
st.set_page_config(
    page_title="Stock Data Analyzer", 
    layout="wide", 
    page_icon=":office:",
    menu_items={
        'About': "Apps ini dibuat dengan bantuan dari AI. Menggunakan groq dan sumber dari Y Finance\n"
                 "mau buat yang sama juga, bisa chat saya ya"
    }
)

st.title("Kucing Abu abu Baik Hati")
st.write("AI powered stock data analyzer")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Analyze", "Stock Data", "Financial Statement","Chat with Stock","News"])

# Sidebar for inputs
with st.sidebar:
    clear_cache_button = st.button("Clear Session Cache")
    symbol = st.text_input(
        "Enter stock symbol:",
        placeholder="input stock symbol, ex: BBCA.JK (from indonesia exchange), ADBE",
    ).upper()
    start_date = st.date_input("Start Date", value=pd.to_datetime("2024-01-01"))
    end_date = st.date_input("End Date", value=pd.Timestamp.now())
    extra_prompt = st.text_area(
        "Tulis hal yang ingin ditanyakan:",
        placeholder="Anda bisa menuliskan hal yang mau ditanyakan berkaitan dengan saham yang ingin Anda analisa, misal 'Berapa titik support untuk saham'",
        height=200
    )
    analyze_button = st.button("Fetch and Analyze Data")

if clear_cache_button:
    for key in st.session_state.keys():
        del st.session_state[key]
    st.experimental_rerun()

if analyze_button or 'symbol' in st.session_state:
    if analyze_button:
        st.session_state.symbol = symbol
        st.session_state.start_date = start_date
        st.session_state.end_date = end_date
        st.session_state.extra_prompt = extra_prompt

        with st.spinner('Fetching and analyzing data...'):
            stock_data = yf.download(st.session_state.symbol, start=st.session_state.start_date, end=st.session_state.end_date)
            ticker_data = yf.Ticker(st.session_state.symbol)
            info = ticker_data.info

            technical_data = stock_data[["Open", "Close"]].to_string()
            fundamental_data = extract_specific_fundamental_data(info)
            fundamental_data_str = format_specific_fundamental_data(fundamental_data)
            
            income_stmt = ticker_data.financials
            quarterly_income_stmt = ticker_data.quarterly_financials
            balance_sheet = ticker_data.balance_sheet
            quarterly_balance_sheet = ticker_data.quarterly_balance_sheet
            cashflow = ticker_data.cashflow
            quarterly_cashflow = ticker_data.quarterly_cashflow

            recommendations = ticker_data.recommendations
            recommendations_str = format_recommendations(recommendations)
            recommendations_summary = ticker_data.recommendations_summary
            
            news = ticker_data.news

            try:
                major_shares = ticker_data.major_holders
                major_shares_available = True
            except Exception as e:
                major_shares = None
                major_shares_available = False

            # Store data in session state
            st.session_state.stock_data = stock_data
            st.session_state.technical_data = technical_data
            st.session_state.fundamental_data_str = fundamental_data_str
            st.session_state.income_stmt = income_stmt
            st.session_state.quarterly_income_stmt = quarterly_income_stmt
            st.session_state.balance_sheet = balance_sheet
            st.session_state.quarterly_balance_sheet = quarterly_balance_sheet
            st.session_state.cashflow = cashflow
            st.session_state.quarterly_cashflow = quarterly_cashflow
            st.session_state.recommendations_str = recommendations_str
            st.session_state.recommendations_summary = recommendations_summary
            st.session_state.news = news
            st.session_state.major_shares = major_shares
            st.session_state.major_shares_available = major_shares_available

            # Analyze the stock
            analysis = analyze_stock_data(st.session_state.technical_data, st.session_state.fundamental_data_str, st.session_state.recommendations_str)
            st.session_state.analysis = analysis
            
            # Initialize extra_prompt_analysis if not present
            if 'extra_prompt_analysis' not in st.session_state:
                st.session_state.extra_prompt_analysis = ""
                
            # Handle extra prompt analysis
            if st.session_state.extra_prompt:
                extra_prompt_analysis = analyze_extra_prompt(st.session_state.extra_prompt, analysis)
                st.session_state.extra_prompt_analysis = extra_prompt_analysis

    # Layout for results
    with tab1:
        left_col, right_col = st.columns(2)

        with left_col:
            st.subheader(f"Analysis for {st.session_state.symbol}")
            st.markdown(f'<div class="stock-analysis">{st.session_state.analysis}</div>', unsafe_allow_html=True)

        with right_col:
            st.subheader("Hasil analisa jawaban untuk pertanyaan")
            st.markdown(f'<div class="stock-analysis">{st.session_state.extra_prompt_analysis}</div>', unsafe_allow_html=True)

            st.subheader(f"Stock Data Graph {st.session_state.symbol}")
            fig = go.Figure(data=[go.Candlestick(x=st.session_state.stock_data.index,
                                                open=st.session_state.stock_data['Open'],
                                                high=st.session_state.stock_data['High'],
                                                low=st.session_state.stock_data['Low'],
                                                close=st.session_state.stock_data['Close'])])
            fig.update_layout(title=f"{st.session_state.symbol} Stock Candlestick Chart", xaxis_title="Date", yaxis_title="Price (IDR)")
            st.plotly_chart(fig)

    with tab2:
        st.subheader(f"Stock Data {st.session_state.symbol}")
        st.dataframe(st.session_state.stock_data, width=1000)    

        st.subheader("Stock Information")
        st.markdown(st.session_state.fundamental_data_str)        

        st.subheader("Financial Statements")
        st.metric(label="Income Statement", value="Income Statement")
        st.dataframe(st.session_state.income_stmt)

        st.metric(label="Quarterly Income Statement", value="Quarterly Income Statement")
        st.dataframe(st.session_state.quarterly_income_stmt)

        st.metric(label="Balance Sheet", value="Balance Sheet")
        st.dataframe(st.session_state.balance_sheet)

        st.metric(label="Quarterly Balance Sheet", value="Quarterly Balance Sheet")
        st.dataframe(st.session_state.quarterly_balance_sheet)

        st.metric(label="Cash Flow Statement", value="Cash Flow Statement")
        st.dataframe(st.session_state.cashflow)

        st.metric(label="Quarterly Cash Flow Statement", value="Quarterly Cash Flow Statement")
        st.dataframe(st.session_state.quarterly_cashflow)

    with tab3:
        st.subheader("Total Revenue per Quarter")
        if 'Total Revenue' in st.session_state.quarterly_income_stmt.index:
            total_revenue = st.session_state.quarterly_income_stmt.loc['Total Revenue'].dropna()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=total_revenue.index, y=total_revenue.values, name='Total Revenue'))
            fig.update_layout(title=f"{st.session_state.symbol} Total Revenue per Quarter", xaxis_title="Quarter", yaxis_title="Amount (IDR)")
            st.plotly_chart(fig)
        else:
            st.write("Row 'Total Revenue' not found in the quarterly income statement.")

    with tab4:
        st.subheader("Chat with Data")

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        user_input = st.text_input("Tuliskan apa yang mau ditanyakan disini", key="chat")

        if user_input:
            yahoo_connector = YahooFinanceConnector(st.session_state.symbol)
            inputPrompt = f"Jawablah pertanyaan {user_input} dengan menggunakan data {yahoo_connector} dari saham {st.session_state.symbol}, jawab dengan kata yang jelas langsung ke kesimpulan tanpa analisa yang terlalu panjang. jawab dengan maksimal 100 kata "
            response = llm.complete(inputPrompt)
            
            # Log the response for debugging
            #logging.debug(f"Response from chat: {response}")

            #st.session_state.chat_history.append({"user": user_input, "bot": response})
            st.markdown(response)

    with tab5:
        st.subheader(f"News related to {st.session_state.symbol}")
        stock = str(st.session_state.symbol)
        
        if stock.endswith(".JK"):
            stock = stock.replace(".JK", "")

        results = DDGS().news(keywords=stock, region="id-en", safesearch="off", timelimit="w", max_results=10)
        newsResultPrompt = f"Buat dalam bahasa INDONESIA DALAM BENTUK NARASI. TIDAK BOLEH PER POIN: berdasarkan hasil dari list body{results}, pastikan bahwa berita yang diambil sesuai dengan {stock},bila tidak sesuai, maka tuliskan tidak ada berita yang sesuai dengan saham yang diminta analisa berita yang bernilai sentimen positif, netral dan negatif. dari hasil analisa tersebut, buatkan laporan dengan bentuk narasi dengan merangkum konten dari masing - masing berita per sentimen dengan bahasa yang jelas dan informatif Berikan link untuk sumber berita. "
        newsResult = llm.complete(newsResultPrompt)
        st.markdown(newsResult)        


        