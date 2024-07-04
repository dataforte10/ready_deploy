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

# Load environment variables
load_dotenv()

# Load Google API key from environment
GROOQ_API_KEY = os.getenv("GROOQ_API_KEY")

# Initialize Gemini
llm = Groq(model="llama3-70b-8192", api_key=GROOQ_API_KEY)

# Function to analyze stock data
def analyze_stock_data(technical_data, fundamental_data):
    # Create a prompt template for analysis
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
    )
    prompt = PromptTemplate(template)

    # Format the prompt with the data
    formatted_prompt = prompt.format(technical_data=technical_data, fundamental_data=fundamental_data)
    
    # Get the analysis from Gemini
    response = llm.complete(formatted_prompt)

    return response

def analyze_extra_prompt(extra_prompt,analysis):
    template = (
    "Sebagai seorang analis saham, berdasarkan data dari {analysis}, jawab pertanyaan yang diajukan oleh user ini: {extra_prompt}. Jawaban harus informatif dan tidak bertele-tele. Apabila Anda tidak memiliki jawabannya, bisa Anda tuliskan 'Saya tidak memiliki data yang cukup.'"
    )
    prompt = PromptTemplate(template)
    formatted_prompt = prompt.format(analysis=analysis, extra_prompt=extra_prompt)
    extra_prompt_response = llm.complete(formatted_prompt)
    
    return extra_prompt_response

# Extract specific fundamental data
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

# Format specific fundamental data for display
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

# Function to load Lottie animation
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
    

# Streamlit Interface
st.set_page_config(
    page_title="Stock Data Analyzer", 
    layout="wide", 
    page_icon=":office:",
    menu_items ={
        'About': "Apps ini dibuat dengan bantuan dari AI. Menggunakan groq dan sumber dari Y Finance\n" 
        "mau buat yang sama juga, bisa chat saya ya"
        } )
st.title("Kucing Abu abu Baik Hati")
st.write("AI powered stock data analyzer")
tab1, tab2, tab3, tab4 = st.tabs(["Analyze", "Stock Data", "Financial Statement","Major Shares"])

# Sidebar for inputs
with st.sidebar:
    symbol = st.text_input(
        "Enter stock symbol:",
        placeholder = "input stock symbol, ex: BBCA.JK (from indonesia exchange), ADBE",
        ).upper()
    start_date = st.date_input("Start Date", value=pd.to_datetime("2024-01-01"))
    end_date = st.date_input("End Date", value=pd.Timestamp.now())
    # Text area for additional questions or prompts
    extra_prompt = st.text_area(
    "Tulis hal yang ingin ditanyakan:",
    placeholder="Anda bisa menuliskan hal yang mau ditanyakan berkaitan dengan saham yang ingin Anda analisa, misal 'Berapa titik support untuk saham'",
    height=200
    )
    analyze_button = st.button("Fetch and Analyze Data")

if analyze_button:
    with tab1:
        # Show loading animation
        with st.spinner('Fetching and analyzing data...'):
            # Fetch stock data
            stock_data = yf.download(symbol, start=start_date, end=end_date)
            ticker_data = yf.Ticker(symbol)
            info = ticker_data.info

            # Prepare data for analysis
            technical_data = stock_data[["Open", "Close"]].to_string()
            fundamental_data = extract_specific_fundamental_data(info)
            fundamental_data_str = format_specific_fundamental_data(fundamental_data)

            # Fetch financial statements
            income_stmt = ticker_data.financials
            quarterly_income_stmt = ticker_data.quarterly_financials
            balance_sheet = ticker_data.balance_sheet
            quarterly_balance_sheet = ticker_data.quarterly_balance_sheet
            cashflow = ticker_data.cashflow
            quarterly_cashflow = ticker_data.quarterly_cashflow
            try:
                major_shares = ticker_data.major_holders
                major_shares_available = True
            except Exception as e:
                major_shares = None
                major_shares_available = False

            # Layout for results
            left_col, right_col = st.columns(2)

        with left_col:
            # Analyze the stock
            st.subheader(f"Analysis for {symbol}")
            analysis = analyze_stock_data(technical_data, fundamental_data_str)
            st.markdown(f'<div class="stock-analysis">{analysis}</div>', unsafe_allow_html=True)

        with right_col:
            # Display extra promp analysis
            st.subheader("Hasil analisa jawaban untuk pertanyaan")
            extra_prompt_analysis = analyze_extra_prompt(extra_prompt, analysis)
            st.markdown(f'<div class="stock-analysis">{extra_prompt_analysis}</div>', unsafe_allow_html=True)


            # Display stock data as a candlestick chart
            st.subheader(f"Stock Data Graph {symbol}")
            fig = go.Figure(data=[go.Candlestick(x=stock_data.index,
                                                open=stock_data['Open'],
                                                high=stock_data['High'],
                                                low=stock_data['Low'],
                                                close=stock_data['Close'])])
            fig.update_layout(title=f"{symbol} Stock Candlestick Chart", xaxis_title="Date", yaxis_title="Price (IDR)")
            st.plotly_chart(fig)
            
        with tab2: 
            # Display stock data
            st.subheader(f"Stock Data {symbol}")
            st.dataframe(stock_data)

            # Display specific fundamental information
            st.subheader("Stock Information")
            st.markdown(fundamental_data_str)        

            # Display financial metrics using st.metric
            st.subheader("Financial Statements")
            st.metric(label="Income Statement", value="Income Statement")
            st.dataframe(income_stmt)

            st.metric(label="Quarterly Income Statement", value="Quarterly Income Statement")
            st.dataframe(quarterly_income_stmt)

            st.metric(label="Balance Sheet", value="Balance Sheet")
            st.dataframe(balance_sheet)

            st.metric(label="Quarterly Balance Sheet", value="Quarterly Balance Sheet")
            st.dataframe(quarterly_balance_sheet)

            st.metric(label="Cash Flow Statement", value="Cash Flow Statement")
            st.dataframe(cashflow)

            st.metric(label="Quarterly Cash Flow Statement", value="Quarterly Cash Flow Statement")
            st.dataframe(quarterly_cashflow)

        with tab3:
            # Plot Total Revenue per quarter
            st.subheader("Total Revenue per Quarter")
            if 'Total Revenue' in quarterly_income_stmt.index:
                total_revenue = quarterly_income_stmt.loc['Total Revenue'].dropna()
                fig = go.Figure()
                fig.add_trace(go.Bar(x=total_revenue.index, y=total_revenue.values, name='Total Revenue'))
                fig.update_layout(title=f"{symbol} Total Revenue per Quarter", xaxis_title="Quarter", yaxis_title="Amount (IDR)")
                st.plotly_chart(fig)
            else:
                st.write("Row 'Total Revenue' not found in the quarterly income statement.")

        with tab4:
            st.subheader("Major Shareholders")
            if major_shares_available:
                st.dataframe(major_shares)
                insiders_percent = major_shares.loc[major_shares[0] == 'insidersPercentHeld', 'Value']
                institutions_percent = major_shares.loc[major_shares[0] == 'institutionsPercentHeld', 'Value']
                float_percent = major_shares.loc[major_shares[0] == 'institutionsFloatPercentHeld', 'Value']

                labels = ['Insiders', 'Institutions', 'Float']
                values = [insiders_percent, institutions_percent, float_percent]

                fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
                fig.update_layout(title=f"{symbol} Major Shareholders")
                st.plotly_chart(fig)
            else:
                st.write("Major shareholders' data is not available.")
