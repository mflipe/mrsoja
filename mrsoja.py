from re import I
import psycopg2
import numpy as np

# from numpy import ones, vstack
# from numpy.linalg import lstsq

import plotly.graph_objects as go
import pandas as pd
import pandas.io.sql as sqlio
import streamlit as st
import datetime
import math

# from datetime import date

st.set_page_config(page_title="Estudo da Soja", layout="wide", page_icon="🌱")  #


def mediaMovel(column, nPeriodos, tipo):
    if tipo == "ari":
        newColumn = (
            column.rolling(window=nPeriodos, min_periods=nPeriodos).mean().round(2)
        )
    elif tipo == "exp":
        newColumn = column.ewm(span=nPeriodos).mean().round(2)
    else:
        return None

    return newColumn


# Initialize connection.
# Uses st.cache to only run once.
@st.cache(allow_output_mutation=True, hash_funcs={"_thread.RLock": lambda _: None})
def init_connection():
    return psycopg2.connect(**st.secrets["postgres"])


conn = init_connection()


def gerarDados():
    df = sqlio.read_sql_query("SELECT * from soja_bmf;", conn)

    # Extraction
    data = df[["Data", "À vista R$", "À vista US$"]].copy()

    # Transformation
    diario = calcularMM(data, "À vista R$")

    semanal = sqlio.read_sql_query("SELECT * from soja_bmf_semanal;", conn)
    semanal["Data"] = pd.to_datetime(semanal["Data"])

    mensal = sqlio.read_sql_query("SELECT * from soja_bmf_mensal;", conn)
    mensal["Data"] = pd.to_datetime(mensal["Data"])

    anual = sqlio.read_sql_query("SELECT * from soja_bmf_anual;", conn)
    anual["Data"] = pd.to_datetime(anual["Data"])

    return diario, semanal, mensal, anual


def calcularMM(data, column):
    # Transformation
    data["MME9"] = mediaMovel(data[column], 9, "exp")
    data["MME80"] = mediaMovel(data[column], 80, "exp")
    data["MME400"] = mediaMovel(data[column], 400, "exp")

    data["MMA21"] = mediaMovel(data[column], 21, "ari")
    data["MMA51"] = mediaMovel(data[column], 51, "ari")
    data["MMA200"] = mediaMovel(data[column], 200, "ari")
    data["MMA400"] = mediaMovel(data[column], 400, "ari")

    return data


@st.cache
def reAmostragem(df, periodo="M"):
    df3 = df.copy()
    df3.index = pd.to_datetime(df3["Data"], infer_datetime_format=True)

    df4 = pd.DataFrame()

    df4["MAX"] = df3["À vista R$"].resample(periodo).max()
    df4["MIN"] = df3["À vista R$"].resample(periodo).min()
    df4["OPEN"] = df3["À vista R$"].resample(periodo).first()
    df4["À vista R$"] = df3["À vista R$"].resample(periodo).last()
    df4 = calcularMM(df4, "À vista R$")
    df4.reset_index(inplace=True)
    df4 = df4.rename(columns={"index": "Data"})

    return df4


def gerarGraficoPlotly(
    df, tipoGrafico, parametros, line_width, candleColor, flag1=False
):
    layout = go.Layout(
        plot_bgcolor="#FFF",  # Sets background color to white
        hovermode="x",  #
        hoverdistance=100,  # Distance to show hover label of data point
        spikedistance=1000,  # Distance to show spike
        xaxis=dict(
            title="Data",
            linecolor="#BCCCDC",  # Sets color of X-axis line
            showgrid=parametros["showgrid"],  # Removes X-axis grid lines
            showspikes=True,  # Show spike line for X-axis
            gridcolor="#BCCCDC",
            # Format spike
            spikethickness=2,
            spikedash="dot",
            spikecolor="#999999",
            spikemode="across",
            fixedrange=parametros["fixedrange"],
            spikesnap="cursor",
        ),
        yaxis=dict(
            title="Preço (R$)",
            linecolor="#BCCCDC",  # Sets color of Y-axis line
            showgrid=parametros["showgrid"],  # Removes Y-axis grid lines
            gridcolor="#BCCCDC",
            showspikes=True,  # Show spike line for X-axis
            # Format spike
            spikethickness=2,
            spikedash="dot",
            spikecolor="#999999",
            spikemode="across",
            fixedrange=True,
            side="right",
            spikesnap="cursor",
        ),
        margin=go.layout.Margin(
            l=0,  # left margin
            r=0,  # right margin
            b=0,  # bottom margin
            t=0,  # top margin
        ),
    )

    df2 = df.copy()

    if flag1 == True:
        df2.index = df2.index + 1
        df2.reset_index(drop=True)

        df3 = pd.DataFrame()
        df3["A"] = df["À vista R$"]
        df3["B"] = df2["À vista R$"]
        df3["high"] = df3[["A", "B"]].max(axis=1)
        df3["low"] = df3[["A", "B"]].min(axis=1)
    else:

        df3 = pd.DataFrame()
        df3["A"] = df["À vista R$"]
        df3["B"] = df["OPEN"]
        df3["high"] = df["MAX"]
        df3["low"] = df["MIN"]

    if tipoGrafico == "CandleStick":
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df["Data"],
                    open=df3["B"],
                    high=df3["high"],
                    low=df3["low"],
                    close=df3["A"],
                    name="À vista R$",
                    increasing_line_color=candleColor[0],
                    decreasing_line_color=candleColor[1],
                )
            ],
            layout=layout,
        )
        fig.update_layout(xaxis_rangeslider_visible=False)
    else:
        fig = go.Figure(
            go.Scatter(
                x=df["Data"],
                y=df["À vista R$"],
                name="À vista R$",
                line=dict(color="Black", width=line_width[0]),
            ),
            layout=layout,
        )

    colors = {
        "MME9": "#FF00FF",
        "MME80": "#FF4500",
        "MME400": "#FF0000",
        "MMA21": "#FFA000",
        "MMA51": "#008000",
        "MMA200": "#008000",
        "MMA400": "#FF5050",
    }

    for key in mediasmoveis:
        if mediasmoveis[key] is True:
            fig.add_trace(
                go.Scatter(
                    x=df["Data"],
                    y=df[key],
                    name=key,
                    line=dict(color=colors[key], width=line_width[1]),
                )
            )

    fig.update_layout(
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0, 0, 0, 0.025)",  # Legend Backgound color
        )
    )

    # build complete timepline from start date to end date
    dt_all = pd.date_range(start=df["Data"].iloc[0], end=df["Data"].iloc[-1])
    # retrieve the dates that ARE in the original datset
    dt_obs = [d.strftime("%Y-%m-%d") for d in pd.to_datetime(df["Data"])]
    # define dates with missing values
    dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d").tolist() if not d in dt_obs]
    fig.update_xaxes(
        rangebreaks=[
            # dict(bounds=["sat", "mon"]),  # hide weekends
            dict(values=dt_breaks)
        ]
    )

    if parametros["hoverinfo"]:
        fig.update_layout(hoverdistance=0)
        fig.update_traces(xaxis="x", hoverinfo="none")

    if parametros["SuporteResistencia"]:
        h_lines = [
            179.30,
            168.20,
            162.96,
            147.30,
            170.97,
            102.75,
            116.27,
            84.73,
            90.97,
            83.41,
            73.30,
        ]

        for line in h_lines:
            if not line <= (df["À vista R$"].min() * 0.95):
                fig.add_hline(
                    y=line,
                    # line_dash="dot",
                    annotation_text="{:.2f}".format(line),
                    annotation_position="bottom",
                    line_width=5,
                    opacity=0.25,
                )

    if parametros["CalendarioAgricola"]:
        anos = range(df["Data"].min().year, df["Data"].max().year, 1)

        if len(anos) == 0:
            anos = [df["Data"].min().year - 1]

        for ano in anos:
            df_aux = df2[
                (df2["Data"] >= pd.to_datetime(datetime.datetime(ano, 9, 1)))
                & (df2["Data"] <= pd.to_datetime(datetime.datetime(ano + 1, 1, 1)))
            ].copy()

            if len(df_aux) > 0:
                print(df_aux["Data"].max())
                if df_aux["Data"].max() > pd.to_datetime(
                    datetime.datetime(ano + 1, 1, 1)
                ):
                    print("IF")
                    fig.add_vrect(
                        x0=df_aux["Data"].min(),
                        x1=datetime.datetime(ano + 1, 1, 1),
                        line_width=0,
                        fillcolor="brown",
                        opacity=0.10,
                        annotation_text="Plantio",
                        annotation_position="bottom left",
                    )
                else:
                    print("Else")
                    fig.add_vrect(
                        x0=df_aux["Data"].min(),
                        x1=df_aux["Data"].max(),
                        line_width=0,
                        fillcolor="brown",
                        opacity=0.10,
                        annotation_text="Plantio",
                        annotation_position="bottom left",
                    )

            df_aux = df2[
                (df2["Data"] >= pd.to_datetime(datetime.datetime(ano + 1, 2, 1)))
                & (df2["Data"] <= pd.to_datetime(datetime.datetime(ano + 1, 5, 1)))
            ].copy()

            if len(df_aux) > 0:
                if df_aux["Data"].max() > pd.to_datetime(
                    datetime.datetime(ano + 1, 5, 1)
                ):
                    fig.add_vrect(
                        x0=df_aux["Data"].min(),
                        x1=datetime.datetime(2021, 5, 1),
                        line_width=0,
                        fillcolor="yellow",
                        opacity=0.10,
                        annotation_text="Colheita",
                        annotation_position="bottom left",
                    )
                else:
                    fig.add_vrect(
                        x0=df_aux["Data"].min(),
                        x1=df_aux["Data"].max(),
                        line_width=0,
                        fillcolor="yellow",
                        opacity=0.10,
                        annotation_text="Colheita",
                        annotation_position="bottom left",
                    )

    # fig.update_annotations(
    #    # textangle=-90,
    #    font_family="Droid Sans Mono",
    #    font_size=20,
    # )

    if parametros["LT"]:
        LT = [
            [
                datetime.datetime(2020, 5, 15),
                116.27,
                datetime.datetime(2020, 6, 6),
                108.25,
            ],
            [
                datetime.datetime(2020, 6, 9),
                102.75,
                datetime.datetime(2021, 3, 21),
                168.17,
            ],
        ]

        for i in LT:
            # Linha de Tendencia está no periodo?
            fig = plotLinhaTendencia(fig, i, df, dt_breaks)

    fig.update_xaxes(
        range=[
            pd.to_datetime(df["Data"].min()) + datetime.timedelta(-1),
            pd.to_datetime(df["Data"].max()) + datetime.timedelta(1),
        ]
    )
    fig.update_yaxes(
        range=[
            int(math.floor(df["À vista R$"].min() / 5.0)) * 5,
            int(math.ceil(df["À vista R$"].max() / 5.0)) * 5,
        ]
    )

    return fig


def plotLinhaTendencia(fig, LT, df, dt_breaks):
    pontoA = {"x": LT[0], "y": LT[1]}
    pontoB = {"x": LT[2], "y": LT[3]}

    points = [
        (0, pontoA["y"]),
        ((pontoB["x"] - pontoA["x"]).days, pontoB["y"]),
    ]

    x_coords, y_coords = zip(*points)
    # A = vstack([x_coords, ones(len(x_coords))]).T
    # m, c = lstsq(A, y_coords, rcond=-1)[0]

    eq = np.polyfit(x_coords, y_coords, 1)
    p = np.poly1d(eq)

    fim = (
        (pontoB["x"] - pontoA["x"]).days
        + abs((LT[2] - pd.to_datetime(df["Data"].max())).days)
        + 1
    )

    pontos = range(0, fim, 1)

    pontosX = np.array([pontoA["x"] + datetime.timedelta(ponto) for ponto in pontos])
    valores = np.array([p(ponto) for ponto in pontos])

    # d = {"X": pontosX, "Y": valores}
    # dataframe_plot = pd.DataFrame(d)
    # f = filter_rows_by_values(dataframe_plot, "X", dt_breaks)
    # print(f)

    fig.add_trace(
        go.Scatter(
            x=[pontosX[0], pontosX[-1]],
            y=[valores[0], valores[-1]],
            name="Linha de Tendência",
            line=dict(color="rgba(0, 0, 0, 0.25)", width=5),
        )
    )

    # fig.update_layout(
    #     shapes=[
    #         dict(
    #             type="line",
    #             yref="y",
    #             xref="x",
    #             x0=pontosX[1],
    #             y0=valores[0],
    #             x1=pontosX[-1],
    #             y1=valores[-1],
    #             line=dict(color="rgba(0, 150, 255, 0.75)", width=5),
    #             layer="below",
    #         ),
    #     ],
    # )

    return fig


def filter_rows_by_values(df, col, values):
    return df[df[col].isin(values) == False]


if __name__ == "__main__":

    #!############################################################################################
    #! HEADER
    #!############################################################################################

    st.title("Estudo para Comercialização da Soja")
    st.subheader("por Marcos Rocha")

    #!############################################################################################
    #!   Sidebar
    #!############################################################################################

    raw_html = """
    <div class="textwidget custom-html-widget">
        <iframe
            frameborder="0"
            src="https://ssltsw.forexprostools.com?lang=12&forex=2103,1617,1513,1,3,9,10&commodities=8918,8914,8919,954867,961618,8916,8915&indices=23660,166,172,27,179,170,174&stocks=474,446,345,346,347,348,469&tabs=1,2,4,3"
            width="400"
            height="467"
            allowtransparency="true"
            marginwidth="0"
            marginheight="0"
            align="center"
        ></iframe>
    </div>
    """
    st.sidebar.markdown(raw_html, unsafe_allow_html=True)

    st.sidebar.title("Menu de Configurações")

    st.sidebar.header("Configurações do Gráfico")

    parametros = {
        "showgrid": True,
        "hoverinfo": True,
        "fixedrange": True,
        "SuporteResistencia": False,
    }
    parametros["showgrid"] = st.sidebar.checkbox("Linhas de Grade", False)
    parametros["hoverinfo"] = not st.sidebar.checkbox("Mostrar Valores", False)
    parametros["fixedrange"] = not st.sidebar.checkbox("Zoom Interativo", False)
    parametros["SuporteResistencia"] = st.sidebar.checkbox(
        "Linhas de Suporte e Resistência", False
    )
    parametros["LT"] = st.sidebar.checkbox("Linhas de Tendência", False)
    parametros["CalendarioAgricola"] = st.sidebar.checkbox("Calendário Agrícola", False)
    if parametros["CalendarioAgricola"]:
        st.sidebar.warning(
            "Calendário Agrícola da Soja para o estado de São Paulo, outros estados somente serão adicionados em caso de solicitação.\
            \n Em Marrom: Plantio\
            \n Em Verde: Maturação\
            \n Em Amarelo: Colheita "
        )

    tipoGrafico = st.sidebar.radio("Tipo", ["CandleStick", "Linha"])
    periodoAmostral = st.sidebar.radio(
        "Periodo", ["Diário", "Semanal", "Mensal", "Anual"]
    )

    candleColor = ["#2ca02c", "#d62728"]
    if tipoGrafico == "CandleStick":
        candleColor[0] = st.sidebar.color_picker(
            "Cor do Candle de Alta", candleColor[0]
        )
        candleColor[1] = st.sidebar.color_picker(
            "Cor do Candle de Baixa", candleColor[1]
        )
        if st.sidebar.button("Reset Cor"):
            candleColor = ["#2ca02c", "#d62728"]

    line_width = [1.0, 0.5, 0.5]
    line_width[0] = st.sidebar.slider(
        "Espessura da Linha Principal", 0.0, 5.0, 1.0, step=0.1
    )
    line_width[1] = st.sidebar.slider(
        "Espessura das Linhas de Média", 0.0, 5.0, 0.5, step=0.1
    )
    line_width[2] = st.sidebar.slider(
        "Espessura das Linhas de Tendência", 0.0, 5.0, 5.0, step=0.1
    )
    if st.sidebar.button("Reset"):
        line_width = [1.0, 0.5, 0.5]

    st.sidebar.subheader("Médias Móveis Exponenciais")
    mme9 = st.sidebar.checkbox("MME9", True)
    mme80 = st.sidebar.checkbox("MME80", True)
    mme400 = st.sidebar.checkbox("MME400", True)

    st.sidebar.subheader("Médias Móveis Aritiméticas")
    mma21 = st.sidebar.checkbox("MMA21", True)
    mma51 = st.sidebar.checkbox("MMA51", True)
    mma200 = st.sidebar.checkbox("MMA200", True)
    mma400 = st.sidebar.checkbox("MMA400", True)

    mediasmoveis = {
        "MME9": mme9,
        "MME80": mme80,
        "MME400": mme400,
        "MMA21": mma21,
        "MMA51": mma51,
        "MMA200": mma200,
        "MMA400": mma400,
    }

    #!############################################################################################
    #! DADOS
    #!############################################################################################
    diario, semanal, mensal, anual = gerarDados()

    df = diario.copy()
    df2 = df.copy()
    df2["Data"] = pd.to_datetime(df["Data"], infer_datetime_format=True).dt.date

    data_min = df2["Data"].min()
    data_max = df2["Data"].max()

    st.subheader("Período")

    col1, col2 = st.columns(2)  # Colunas

    data_inicial = col1.date_input(
        "Data Inicial",
        value=data_max - datetime.timedelta(30),
        min_value=data_min,
        max_value=data_max,
    )
    data_final = col2.date_input(
        "Data Final",
        value=data_max,
        min_value=data_min,
        max_value=data_max,
    )

    droplist = []
    for key in mediasmoveis:
        if mediasmoveis[key] is False:
            droplist.append(key)

    if droplist:
        df2 = df2.drop(columns=droplist)

    #!############################################################################################
    #! body
    #!############################################################################################

    # col1, col2 = st.beta_columns(2) # Colunas
    st.header("Preços Históricos da Soja")

    st.subheader("Tabela de Preços - Saca de 60kg")

    df_table = df2.copy()
    df_table["Dólar"] = df_table["À vista R$"] / df_table["À vista US$"]
    df_table.rename({"À vista R$": "R$", "À vista US$": "US$"}, axis=1, inplace=True)

    col1.subheader("Tabela das últimas medições")

    col1.dataframe(
        df_table[["Data", "R$", "US$", "Dólar"]]
        .sort_index(ascending=False)
        .set_index("Data")
        .style.format({"R$": "{:.2f}", "US$": "{:.2f}", "Dólar": "{:.4}"})
        .set_properties(
            **{
                "background-color": "white",
                "color": "black",
                # "border-color": "white",
            }
        ),
        height=150,
        # width=950,
    )

    col2.subheader("Variação das últimas 7 medições")

    col2.metric(
        label="Soja (R$)",
        value=df_table["R$"].iloc[-1].round(2),
        delta=((df_table["R$"].iloc[-1] / df_table["R$"].iloc[-7] - 1) * 100).round(2),
    )

    col2.metric(
        label="Dólar (US$)",
        value=df_table["Dólar"].iloc[-1].round(2),
        delta=(
            (df_table["Dólar"].iloc[-1] / df_table["Dólar"].iloc[-7] - 1) * 100
        ).round(2),
    )

    # "Periodo", ["Diário", "Semanal", "Mensal", "Anual"]
    if periodoAmostral == "Diário":
        st.subheader("Gráfico de Preços - 1D")
        df2 = df2[
            (df2["Data"] >= pd.to_datetime(data_inicial))
            & (df2["Data"] <= pd.to_datetime(data_final))
        ]
        diario = gerarGraficoPlotly(
            df2, tipoGrafico, parametros, line_width, candleColor, True
        )
        st.plotly_chart(
            diario,
            use_container_width=True,
        )

    if periodoAmostral == "Semanal":
        st.subheader("Gráfico de Preços - 1S")
        df3 = semanal.copy()
        df3 = df3[
            (df3["Data"] >= pd.to_datetime(data_inicial))
            & (df3["Data"] <= pd.to_datetime(data_final))
        ]
        grafico_semanal = gerarGraficoPlotly(
            df3, tipoGrafico, parametros, line_width, candleColor
        )
        st.plotly_chart(grafico_semanal, use_container_width=True)

    if periodoAmostral == "Mensal":
        st.subheader("Gráfico de Preços - 1M")
        df4 = mensal.copy()
        df4 = df4[
            (df4["Data"] >= pd.to_datetime(data_inicial))
            & (df4["Data"] <= pd.to_datetime(data_final))
        ]
        mensal = gerarGraficoPlotly(
            df4, tipoGrafico, parametros, line_width, candleColor
        )
        st.plotly_chart(mensal, use_container_width=True)

    if periodoAmostral == "Anual":
        try:
            st.subheader("Gráfico de Preços - 1Y")
            df4 = anual.copy()

            df4 = df4[
                (df4["Data"] >= pd.to_datetime(data_inicial))
                & (df4["Data"] <= pd.to_datetime(data_final))
            ]
            anual = gerarGraficoPlotly(
                df4, tipoGrafico, parametros, line_width, candleColor
            )
            st.plotly_chart(anual, use_container_width=True)
        except:
            st.warning("Aumente o periodo para pelo menos 1 ano")

    #
    # FOOTER
    #

    footer = """
    Me envie um e-mail me contando como esta ferramenta te ajuda ou sugestões de novas funcionalidades!

    [![NAME Badge](https://img.shields.io/badge/%C2%A9_Marcos_Rocha-2021-red?&style=for-the-badge)](https://www.linkedin.com/in/marcosfeliperocha/)

    [![LINKEDIN Badge](https://img.shields.io/badge/LinkedIn--blue?style=social&logo=LinkedIn&link=https://www.linkedin.com/in/marcosfeliperocha/)](https://www.linkedin.com/in/marcosfeliperocha/)
    [![MAIL Badge](https://img.shields.io/badge/Email-marcos.fellps@gmail.com-c14438?style=plastic&logo=Gmail&logoColor=white&link=mailto:marcos.fellps@gmail.com)](mailto:marcos.fellps@gmail.com)
    [![BLOG Badge](https://img.shields.io/badge/Blog-https://marcosfellps.wordpress.com/-blue?style=plastic&logo=WordPress&logoColor=white&link=https://marcosfellps.wordpress.com/)](https://marcosfellps.wordpress.com/)

    """

    disclaimer = """As informações aqui contidas não constituem uma oferta ou recomendação para compra ou venda de ações, ou de quaisquer outros valores mobiliários, nem poderá ser entendida como tal em qualquer jurisdição na qual tal solicitação, oferta ou recomendação sejam consideradas ilegais. Tampouco oferece conselhos de investimento, tributários ou legais. Os leitores devem buscar orientação profissional sobre investimentos, impostos e legislação antes de investir. Me isentando assim de responsabilidade sobre quaisquer danos resultantes direta ou indiretamente da utilização das informações aqui contidas."""
    st.sidebar.subheader("Contato do Autor:")
    st.sidebar.write(footer)
    st.subheader("Contato do Autor:")

    footer

    st.header("Disclaimer")
    disclaimer
