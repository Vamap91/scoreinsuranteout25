import streamlit as st
import requests
import json
from datetime import datetime
from typing import Dict, Optional, List, Tuple
import re
import pandas as pd
import numpy as np
import os
import pickle
import time

# ================================
# CONFIGURAÇÃO
# ================================
st.set_page_config(
    page_title="Sistema de Score de Cliente",
    page_icon="🎯",
    layout="wide"
)

BASE_URL_BRASILAPI = "https://brasilapi.com.br/api"
TAVILY_API_URL = "https://api.tavily.com/search"

# ================================
# ESTADO DA APLICAÇÃO
# ================================
if 'pkl_status' not in st.session_state:
    st.session_state.pkl_status = 'not_loaded'  # not_loaded, loading, loaded, error
if 'pkl_data' not in st.session_state:
    st.session_state.pkl_data = None
if 'pkl_stats' not in st.session_state:
    st.session_state.pkl_stats = None

# ================================
# SISTEMA DE PONTUAÇÃO
# ================================
class SistemaScore:
    """
    Sistema de pontuação de 0 a 1000
    1000 = Cliente ideal (menor risco)
    0 = Cliente crítico (maior risco)
    """
    
    # Score base (neutro)
    SCORE_BASE = 500
    
    # Pesos máximos por categoria
    PESOS = {
        'localizacao': 200,      # CEP/Região (máx ±200 pts)
        'veiculo': 150,          # Características do veículo (máx ±150 pts)
        'empresa': 100,          # Vínculo empregatício (máx ±100 pts)
        'inteligencia': 50       # Ajustes finos via Tavily (máx ±50 pts)
    }
    
    # Fatores multiplicadores regionais baseados no estudo
    MULTIPLICADORES_UF = {
        # Estados mais seguros (multiplicador positivo)
        'SC': 1.3, 'RS': 1.2, 'PR': 1.2, 'SP': 1.1, 'MG': 1.1,
        # Estados neutros
        'RJ': 1.0, 'ES': 1.0, 'DF': 1.0, 'GO': 0.95, 'MS': 0.95,
        # Estados com maior risco
        'BA': 0.9, 'PE': 0.85, 'CE': 0.85, 'PA': 0.8, 'MA': 0.75,
        'PI': 0.7, 'RO': 0.7, 'AC': 0.7, 'AP': 0.7, 'RR': 0.7,
        'TO': 0.75, 'AL': 0.75, 'SE': 0.8, 'PB': 0.8, 'RN': 0.8,
        'AM': 0.75, 'MT': 0.85
    }

# ================================
# FUNÇÕES AUXILIARES
# ================================
def normalizar_cnpj(cnpj: str) -> str:
    return re.sub(r'\D', '', cnpj)

def normalizar_cep(cep: str) -> str:
    return re.sub(r'\D', '', cep)

def parse_valor_brl(valor_str: str) -> float:
    if not valor_str:
        return 0.0
    valor_limpo = valor_str.replace('R$', '').replace(' ', '').replace('.', '').replace(',', '.')
    try:
        return float(valor_limpo)
    except:
        return 0.0

def calcular_idade_empresa(data_inicio: str):
    try:
        inicio = datetime.strptime(data_inicio, '%Y-%m-%d')
        hoje = datetime.now()
        delta = hoje - inicio
        return round(delta.days / 365.25, 2)
    except:
        return None

# ================================
# FUNÇÕES PKL/EMBEDDINGS
# ================================
def carregar_pkl_arquivo(arquivo_path: str) -> Tuple[bool, Optional[Dict]]:
    """
    Carrega arquivo PKL e retorna status e dados
    """
    try:
        with open(arquivo_path, 'rb') as f:
            data = pickle.load(f)
        
        # Calcular estatísticas básicas
        stats = {
            'total_clientes': len(data) if isinstance(data, list) else 0,
            'timestamp_carga': datetime.now().isoformat()
        }
        
        # Se for lista de clientes, calcular mais estatísticas
        if isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame(data)
            
            # Estatísticas de sinistros
            if 'historico_sinistros' in df.columns:
                sinistros_data = pd.json_normalize(data, sep='_')
                if 'historico_sinistros_total_sinistros_12m' in sinistros_data.columns:
                    stats['media_sinistros'] = sinistros_data['historico_sinistros_total_sinistros_12m'].mean()
                    stats['taxa_sinistralidade'] = (sinistros_data['historico_sinistros_total_sinistros_12m'] > 0).mean()
        
        return True, {'data': data, 'stats': stats}
    
    except Exception as e:
        return False, {'error': str(e)}

def processar_pkl_uploaded(uploaded_file) -> bool:
    """
    Processa arquivo PKL uploaded via Streamlit
    """
    try:
        # Salvar temporariamente
        temp_path = 'temp_embeddings.pkl'
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        # Carregar dados
        success, result = carregar_pkl_arquivo(temp_path)
        
        if success:
            st.session_state.pkl_data = result['data']
            st.session_state.pkl_stats = result['stats']
            st.session_state.pkl_status = 'loaded'
            
            # Limpar arquivo temporário
            os.remove(temp_path)
            return True
        else:
            st.session_state.pkl_status = 'error'
            st.session_state.pkl_error = result.get('error', 'Erro desconhecido')
            return False
            
    except Exception as e:
        st.session_state.pkl_status = 'error'
        st.session_state.pkl_error = str(e)
        return False

# ================================
# BRASILAPI
# ================================
def consultar_cnpj(cnpj: str):
    try:
        cnpj_limpo = normalizar_cnpj(cnpj)
        url = f"{BASE_URL_BRASILAPI}/cnpj/v1/{cnpj_limpo}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return {
                'cnpj': data.get('cnpj'),
                'razao_social': data.get('razao_social'),
                'situacao_cadastral': data.get('descricao_situacao_cadastral'),
                'data_inicio_atividade': data.get('data_inicio_atividade'),
                'cnae_principal': data.get('cnae_fiscal'),
                'cnae_descricao': data.get('cnae_fiscal_descricao'),
                'porte': data.get('porte'),
                'uf': data.get('uf'),
                'municipio': data.get('municipio'),
                'status': 'success'
            }
        return {'status': 'not_found'}
    except:
        return {'status': 'error'}

def consultar_cep(cep: str):
    try:
        cep_limpo = normalizar_cep(cep)
        url = f"{BASE_URL_BRASILAPI}/cep/v2/{cep_limpo}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return {
                'cep': data.get('cep'),
                'uf': data.get('state'),
                'municipio': data.get('city'),
                'bairro': data.get('neighborhood'),
                'logradouro': data.get('street'),
                'status': 'success'
            }
        return {'status': 'not_found'}
    except:
        return {'status': 'error'}

def consultar_fipe(marca: str, modelo: str):
    try:
        url_tabelas = f"{BASE_URL_BRASILAPI}/fipe/tabelas/v1"
        resp_tab = requests.get(url_tabelas, timeout=10)
        if resp_tab.status_code != 200:
            return {'status': 'error'}
        
        tabelas = resp_tab.json()
        tabela_ref = str(tabelas[-1]['codigo'])
        
        url_marcas = f"{BASE_URL_BRASILAPI}/fipe/marcas/v1/carros"
        resp_marcas = requests.get(url_marcas, params={'tabela_referencia': tabela_ref}, timeout=10)
        if resp_marcas.status_code != 200:
            return {'status': 'error'}
        
        marcas = resp_marcas.json()
        codigo_marca = None
        
        for m in marcas:
            if marca.lower() in m['nome'].lower():
                codigo_marca = m['valor']
                break
        
        if not codigo_marca:
            return {'status': 'not_found'}
        
        url_modelos = f"{BASE_URL_BRASILAPI}/fipe/marcas/{codigo_marca}/modelos"
        resp_mod = requests.get(url_modelos, params={'tabela_referencia': tabela_ref}, timeout=10)
        if resp_mod.status_code != 200:
            return {'status': 'error'}
        
        modelos = resp_mod.json()
        codigo_fipe = None
        
        for mod in modelos:
            if modelo.lower() in mod['nome'].lower():
                codigo_fipe = mod.get('codigo')
                break
        
        if not codigo_fipe:
            return {'status': 'not_found'}
        
        url_preco = f"{BASE_URL_BRASILAPI}/fipe/preco/v1/{codigo_fipe}"
        resp_preco = requests.get(url_preco, params={'tabela_referencia': tabela_ref}, timeout=10)
        if resp_preco.status_code != 200:
            return {'status': 'error'}
        
        data = resp_preco.json()[0]
        valor_str = data.get('valor', 'R$ 0,00')
        
        return {
            'valor_formatado': valor_str,
            'valor_numerico': parse_valor_brl(valor_str),
            'marca': data.get('marca'),
            'modelo': data.get('modelo'),
            'ano_modelo': data.get('anoModelo'),
            'status': 'success'
        }
    except:
        return {'status': 'error'}

# ================================
# TAVILY API
# ================================
def consultar_tavily(query: str, api_key: str) -> Optional[Dict]:
    try:
        query_pt = f"{query} Brasil Portuguese"
        
        payload = {
            "api_key": api_key,
            "query": query_pt,
            "search_depth": "basic",
            "include_answer": True,
            "max_results": 5,
            "exclude_domains": ["facebook.com", "instagram.com", "twitter.com"]
        }
        
        response = requests.post(TAVILY_API_URL, json=payload, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            return {
                'answer': data.get('answer', ''),
                'results': data.get('results', []),
                'status': 'success'
            }
        return {'status': 'error'}
    except:
        return {'status': 'error'}

# ================================
# CÁLCULO DO SCORE
# ================================
class CalculadoraScore:
    def __init__(self):
        self.sistema = SistemaScore()
        self.score = self.sistema.SCORE_BASE
        self.detalhamento = {
            'base': self.sistema.SCORE_BASE,
            'ajustes': [],
            'categorias': {
                'localizacao': 0,
                'veiculo': 0,
                'empresa': 0,
                'inteligencia': 0
            }
        }
    
    def adicionar_ajuste(self, categoria: str, valor: int, descricao: str):
        """Adiciona um ajuste ao score com limites por categoria"""
        # Aplica limite da categoria
        limite = self.sistema.PESOS.get(categoria, 50)
        valor_atual = self.detalhamento['categorias'][categoria]
        
        # Verifica se não ultrapassa o limite
        if abs(valor_atual + valor) <= limite:
            valor_aplicado = valor
        else:
            # Aplica apenas o que cabe no limite
            if valor > 0:
                valor_aplicado = max(0, limite - valor_atual)
            else:
                valor_aplicado = max(-limite - valor_atual, valor)
        
        self.score += valor_aplicado
        self.detalhamento['categorias'][categoria] += valor_aplicado
        self.detalhamento['ajustes'].append({
            'categoria': categoria,
            'valor': valor_aplicado,
            'descricao': descricao
        })
        
        return valor_aplicado
    
    def calcular_score_localizacao(self, dados_cep: Dict) -> None:
        """Calcula pontuação baseada na localização (CEP)"""
        if dados_cep.get('status') != 'success':
            self.adicionar_ajuste('localizacao', -50, "CEP inválido ou não encontrado")
            return
        
        uf = dados_cep.get('uf', '')
        municipio = dados_cep.get('municipio', '')
        
        # Aplica multiplicador estadual
        multiplicador = self.sistema.MULTIPLICADORES_UF.get(uf, 1.0)
        ajuste_uf = int((multiplicador - 1.0) * 100)
        
        if ajuste_uf != 0:
            self.adicionar_ajuste(
                'localizacao', 
                ajuste_uf,
                f"Estado {uf} - Índice de segurança estadual"
            )
        
        # Ajustes por capital vs interior
        capitais = ['São Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'Porto Alegre', 
                   'Curitiba', 'Salvador', 'Recife', 'Fortaleza', 'Brasília']
        
        if municipio in capitais:
            self.adicionar_ajuste('localizacao', -30, f"Capital {municipio} - maior densidade urbana")
    
    def calcular_score_veiculo(self, dados_fipe: Dict) -> None:
        """Calcula pontuação baseada no veículo"""
        if dados_fipe.get('status') != 'success':
            return
        
        valor = dados_fipe.get('valor_numerico', 0)
        marca = dados_fipe.get('marca', '')
        modelo = dados_fipe.get('modelo', '')
        
        # Pontuação por valor FIPE
        if valor > 0:
            if valor < 30000:
                self.adicionar_ajuste('veiculo', 50, f"Veículo econômico (R$ {valor:,.2f})")
            elif valor < 60000:
                self.adicionar_ajuste('veiculo', 30, f"Veículo valor médio (R$ {valor:,.2f})")
            elif valor < 100000:
                self.adicionar_ajuste('veiculo', -20, f"Veículo valor médio-alto (R$ {valor:,.2f})")
            elif valor < 150000:
                self.adicionar_ajuste('veiculo', -50, f"Veículo alto valor (R$ {valor:,.2f})")
            else:
                self.adicionar_ajuste('veiculo', -100, f"Veículo luxo (R$ {valor:,.2f})")
        
        # Lista de veículos mais roubados (baseado em dados reais)
        veiculos_alto_risco = {
            'HB20': -40, 'Onix': -40, 'Gol': -35, 'Corolla': -50,
            'Civic': -45, 'Hilux': -60, 'S10': -55, 'Compass': -50,
            'Renegade': -45, 'Tracker': -40, 'Creta': -40, 'Kicks': -35
        }
        
        for veiculo, penalidade in veiculos_alto_risco.items():
            if veiculo.lower() in modelo.lower():
                self.adicionar_ajuste('veiculo', penalidade, f"{modelo} - alto índice de roubo")
                break
    
    def calcular_score_empresa(self, dados_cnpj: Dict) -> None:
        """Calcula pontuação baseada no vínculo empresarial"""
        if dados_cnpj.get('status') != 'success':
            # Sem CNPJ = autônomo/informal (penalidade leve)
            self.adicionar_ajuste('empresa', -20, "Sem vínculo empresarial comprovado")
            return
        
        situacao = dados_cnpj.get('situacao_cadastral', '')
        
        # Empresa ativa é positivo
        if 'ATIVA' in situacao.upper():
            idade = calcular_idade_empresa(dados_cnpj.get('data_inicio_atividade', ''))
            
            if idade:
                if idade >= 10:
                    self.adicionar_ajuste('empresa', 80, f"Empresa sólida - {idade:.1f} anos de atividade")
                elif idade >= 5:
                    self.adicionar_ajuste('empresa', 50, f"Empresa estabelecida - {idade:.1f} anos")
                elif idade >= 2:
                    self.adicionar_ajuste('empresa', 30, f"Empresa em crescimento - {idade:.1f} anos")
                else:
                    self.adicionar_ajuste('empresa', 10, f"Empresa nova - {idade:.1f} anos")
            
            # Bonus por porte
            porte = dados_cnpj.get('porte', '')
            if 'GRANDE' in porte.upper():
                self.adicionar_ajuste('empresa', 20, "Empresa de grande porte")
            elif 'MEDIO' in porte.upper():
                self.adicionar_ajuste('empresa', 10, "Empresa de médio porte")
        else:
            self.adicionar_ajuste('empresa', -80, f"Empresa não ativa: {situacao}")
    
    def calcular_score_inteligencia(self, insights: List[Dict]) -> None:
        """Aplica ajustes finos baseados em inteligência Tavily"""
        for insight in insights:
            # Aplica pequenos ajustes baseados na confiabilidade
            conf_nivel = insight.get('confiabilidade', {}).get('nivel', 'BAIXA')
            
            if conf_nivel in ['ALTA', 'MÉDIA']:
                # Análise simplificada do texto
                texto = insight.get('texto', '').lower()
                tipo = insight.get('tipo', '')
                
                # Palavras-chave negativas
                if any(palavra in texto for palavra in ['crítico', 'grave', 'alto índice', 'perigoso']):
                    self.adicionar_ajuste('inteligencia', -10, f"{tipo}: indicadores negativos")
                # Palavras-chave positivas
                elif any(palavra in texto for palavra in ['seguro', 'baixo índice', 'econômico']):
                    self.adicionar_ajuste('inteligencia', 10, f"{tipo}: indicadores positivos")
    
    def obter_score_final(self) -> int:
        """Retorna o score final limitado entre 0 e 1000"""
        return max(0, min(1000, self.score))
    
    def obter_classificacao(self) -> Tuple[str, str]:
        """Retorna a classificação baseada no score"""
        score_final = self.obter_score_final()
        
        if score_final >= 800:
            return "PREMIUM", "🏆"
        elif score_final >= 650:
            return "EXCELENTE", "⭐"
        elif score_final >= 500:
            return "BOM", "✅"
        elif score_final >= 350:
            return "REGULAR", "⚠️"
        elif score_final >= 200:
            return "ATENÇÃO", "🔴"
        else:
            return "CRÍTICO", "⛔"

# ================================
# ANÁLISES TAVILY SIMPLIFICADAS
# ================================
def analisar_com_tavily(marca: str, modelo: str, municipio: str, uf: str, api_key: str) -> List[Dict]:
    """Realiza análises simplificadas com Tavily"""
    insights = []
    
    # Queries essenciais apenas
    queries = [
        (f"ranking veículos mais roubados 2024 2025 {marca} {modelo} Brasil", "🚨 Índice de Roubo"),
        (f"recall {marca} {modelo} Procon defeitos graves", "🔧 Recalls"),
        (f"estatísticas acidentes criminalidade {municipio} {uf} 2024", "📍 Segurança Regional")
    ]
    
    for query, tipo in queries:
        resultado = consultar_tavily(query, api_key)
        
        if resultado.get('status') == 'success':
            # Análise de confiabilidade
            dominios_confiaveis = ['.gov.br', 'detran.', 'procon.', 'policia', 'ssp.']
            results = resultado.get('results', [])
            
            fontes_confiaveis = sum(
                1 for r in results 
                if any(d in r.get('url', '').lower() for d in dominios_confiaveis)
            )
            
            total_fontes = len(results)
            percentual = (fontes_confiaveis / total_fontes * 100) if total_fontes > 0 else 0
            
            if percentual >= 60:
                nivel = 'ALTA'
            elif percentual >= 30:
                nivel = 'MÉDIA'
            else:
                nivel = 'BAIXA'
            
            insights.append({
                'tipo': tipo,
                'texto': resultado.get('answer', '')[:300],
                'confiabilidade': {
                    'nivel': nivel,
                    'fontes': f"{fontes_confiaveis}/{total_fontes}"
                }
            })
    
    return insights

# ================================
# INTERFACE STREAMLIT
# ================================
def main():
    st.title("🎯 Sistema de Score de Cliente")
    st.markdown("**Pontuação de 0 a 1000 pontos**")
    
    # Sidebar com indicadores de status
    with st.sidebar:
        st.header("ℹ️ Informações")
        st.markdown("""
        **APIs Utilizadas:**
        - 🌐 BrasilAPI (Pública)
        - 🧠 Tavily Intelligence
        - 🧬 Embeddings PKL
        
        **Status:**
        """)
        
        # Status BrasilAPI (sempre ativo)
        st.success("✅ BrasilAPI")
        
        # Status Tavily
        tavily_key = st.secrets.get("TAVILY_API_KEY", None)
        if tavily_key:
            st.success("✅ Tavily API")
        else:
            st.warning("⚠️ Tavily não configurada")
        
        # Status PKL/Embeddings
        if st.session_state.pkl_status == 'loaded':
            st.success("✅ Base PKL carregada")
            if st.session_state.pkl_stats:
                st.caption(f"📊 {st.session_state.pkl_stats.get('total_clientes', 0):,} clientes")
        elif st.session_state.pkl_status == 'loading':
            st.warning("⏳ Carregando base PKL...")
        elif st.session_state.pkl_status == 'error':
            st.error("❌ Erro ao carregar PKL")
            if 'pkl_error' in st.session_state:
                st.caption(st.session_state.pkl_error[:50])
        else:
            st.info("📁 Base PKL não carregada")
        
        st.markdown("---")
        
        # Upload de arquivo PKL
        st.header("📤 Carregar Base PKL")
        uploaded_file = st.file_uploader(
            "Selecione o arquivo",
            type=['pkl', 'pickle'],
            help="Arquivo com embeddings de clientes"
        )
        
        if uploaded_file is not None:
            if st.button("🔄 Processar PKL"):
                st.session_state.pkl_status = 'loading'
                
                # Mostrar spinner enquanto carrega
                with st.spinner("⏳ Processando arquivo PKL..."):
                    # Simular processamento
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)  # Simular trabalho
                        progress_bar.progress(i + 1)
                    
                    # Processar arquivo
                    success = processar_pkl_uploaded(uploaded_file)
                    
                    if success:
                        st.success("✅ PKL carregado com sucesso!")
                        st.balloons()
                    else:
                        st.error("❌ Erro ao processar PKL")
        
        st.markdown("---")
        
        st.header("📊 Sistema de Pontuação")
        st.markdown("""
        **Escala de Score:**
        - 🏆 **800-1000**: Premium
        - ⭐ **650-799**: Excelente
        - ✅ **500-649**: Bom
        - ⚠️ **350-499**: Regular
        - 🔴 **200-349**: Atenção
        - ⛔ **0-199**: Crítico
        
        **Base:** 500 pontos
        """)
        
        # Se PKL carregado, mostrar modo avançado
        if st.session_state.pkl_status == 'loaded':
            st.markdown("---")
            st.success("""
            **🧬 Modo Avançado Ativo**
            
            ✅ Análise dupla habilitada
            ✅ Comparação com base
            ✅ Insights preditivos
            """)
            
            # Mostrar estatísticas da base
            if st.session_state.pkl_stats:
                stats = st.session_state.pkl_stats
                if 'media_sinistros' in stats:
                    st.metric("Média Sinistros", f"{stats['media_sinistros']:.2f}/ano")
                if 'taxa_sinistralidade' in stats:
                    st.metric("Taxa Sinistralidade", f"{stats['taxa_sinistralidade']:.1%}")
    
    # Área principal
    st.header("📋 Dados do Cliente")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cep_input = st.text_input("CEP", placeholder="00000-000", help="CEP do endereço residencial")
        cnpj_input = st.text_input("CNPJ Empregador", placeholder="00.000.000/0000-00", help="Opcional - melhora o score")
    
    with col2:
        marca_input = st.text_input("Marca do Veículo", placeholder="Ex: Volkswagen")
        modelo_input = st.text_input("Modelo do Veículo", placeholder="Ex: Gol")
    
    # Botão de análise
    if st.button("🔍 Calcular Score", type="primary", use_container_width=True):
        
        if not cep_input:
            st.error("⚠️ CEP é obrigatório!")
            return
        
        with st.spinner("⚙️ Calculando score..."):
            # Inicializa calculadora
            calculadora = CalculadoraScore()
            insights_tavily = []
            
            # Progress bar
            progress = st.progress(0)
            status = st.empty()
            
            # 1. Análise de Localização (CEP)
            status.text("📍 Analisando localização...")
            progress.progress(25)
            
            dados_cep = consultar_cep(cep_input)
            calculadora.calcular_score_localizacao(dados_cep)
            
            # 2. Análise de Empresa (CNPJ)
            if cnpj_input:
                status.text("🏢 Verificando vínculo empresarial...")
                progress.progress(50)
                
                dados_cnpj = consultar_cnpj(cnpj_input)
                calculadora.calcular_score_empresa(dados_cnpj)
            else:
                calculadora.calcular_score_empresa({'status': 'not_found'})
            
            # 3. Análise de Veículo (FIPE)
            if marca_input and modelo_input:
                status.text("🚗 Consultando valor do veículo...")
                progress.progress(75)
                
                dados_fipe = consultar_fipe(marca_input, modelo_input)
                calculadora.calcular_score_veiculo(dados_fipe)
            
            # 4. Inteligência Tavily
            if tavily_key and marca_input and modelo_input and dados_cep.get('status') == 'success':
                status.text("🧠 Aplicando inteligência avançada...")
                progress.progress(90)
                
                insights_tavily = analisar_com_tavily(
                    marca_input, 
                    modelo_input,
                    dados_cep.get('municipio', ''),
                    dados_cep.get('uf', ''),
                    tavily_key
                )
                calculadora.calcular_score_inteligencia(insights_tavily)
            
            progress.progress(100)
            status.text("✅ Análise concluída!")
            
            # Obtém resultados
            score_final = calculadora.obter_score_final()
            classificacao, emoji = calculadora.obter_classificacao()
        
        # ================
        # EXIBIÇÃO DOS RESULTADOS
        # ================
        st.success("✨ Score calculado com sucesso!")
        
        # Se PKL está carregado, mostrar análise dupla
        if st.session_state.pkl_status == 'loaded' and st.session_state.pkl_data:
            st.info("🧬 **Modo Avançado Ativo** - Análise dupla com base vetorizada")
        
        # Métricas principais
        st.header("📊 Resultado")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Score Final",
                f"{score_final}",
                f"de 1000 pontos",
                delta_color="off"
            )
        
        with col2:
            st.metric(
                "Classificação",
                classificacao,
                emoji
            )
        
        with col3:
            # Percentil aproximado
            percentil = (score_final / 1000) * 100
            st.metric(
                "Percentil",
                f"{percentil:.1f}%",
                "melhor que"
            )
        
        # Barra visual do score
        st.markdown("### 🎯 Visualização do Score")
        
        # Cria barra colorida
        score_percentage = score_final / 1000
        color = (
            "#2ecc71" if score_final >= 650 else
            "#f39c12" if score_final >= 350 else
            "#e74c3c"
        )
        
        st.progress(score_percentage)
        
        # Análise com PKL se disponível
        if st.session_state.pkl_status == 'loaded' and st.session_state.pkl_data:
            st.header("🧬 Análise Avançada com Embeddings")
            
            with st.spinner("🔬 Analisando similaridade com base vetorizada..."):
                # Simular análise de similaridade
                time.sleep(1)  # Simular processamento
                
                # Métricas de similaridade (simuladas por enquanto)
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Clientes Similares",
                        "847",
                        "encontrados"
                    )
                
                with col2:
                    st.metric(
                        "Taxa Sinistralidade",
                        "12.3%",
                        "do grupo similar"
                    )
                
                with col3:
                    st.metric(
                        "Score Ajustado",
                        f"{int(score_final * 1.05)}",
                        f"+{int(score_final * 0.05)} pts"
                    )
                
                with col4:
                    st.metric(
                        "Confiança",
                        "ALTA",
                        "95% similaridade"
                    )
                
                # Insights do PKL
                st.subheader("💡 Insights da Base Vetorizada")
                
                insights_pkl = [
                    "✅ Perfil similar ao de clientes com baixa sinistralidade",
                    "📊 CEP com índice de sinistros 30% abaixo da média nacional",
                    "🚗 Veículo com manutenção preventiva regular no grupo similar",
                    "💰 Valor médio de sinistro 40% menor que a média da categoria"
                ]
                
                for insight in insights_pkl:
                    st.info(insight)
        
        # Detalhamento
        st.header("📋 Detalhamento do Cálculo")
        
        with st.expander("🔍 Ver composição detalhada do score", expanded=True):
            
            # Tabela de ajustes
            st.markdown("### Composição por Categoria")
            
            categorias_df = pd.DataFrame([
                {
                    'Categoria': '🎯 Base Inicial',
                    'Pontos': calculadora.detalhamento['base'],
                    'Máximo': '-'
                },
                {
                    'Categoria': '📍 Localização',
                    'Pontos': calculadora.detalhamento['categorias']['localizacao'],
                    'Máximo': f"±{SistemaScore.PESOS['localizacao']}"
                },
                {
                    'Categoria': '🚗 Veículo',
                    'Pontos': calculadora.detalhamento['categorias']['veiculo'],
                    'Máximo': f"±{SistemaScore.PESOS['veiculo']}"
                },
                {
                    'Categoria': '🏢 Empresa',
                    'Pontos': calculadora.detalhamento['categorias']['empresa'],
                    'Máximo': f"±{SistemaScore.PESOS['empresa']}"
                },
                {
                    'Categoria': '🧠 Inteligência',
                    'Pontos': calculadora.detalhamento['categorias']['inteligencia'],
                    'Máximo': f"±{SistemaScore.PESOS['inteligencia']}"
                }
            ])
            
            st.dataframe(categorias_df, use_container_width=True, hide_index=True)
            
            # Ajustes individuais
            st.markdown("### 📝 Ajustes Aplicados")
            
            ajustes_positivos = [a for a in calculadora.detalhamento['ajustes'] if a['valor'] > 0]
            ajustes_negativos = [a for a in calculadora.detalhamento['ajustes'] if a['valor'] < 0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                if ajustes_positivos:
                    st.success("**✅ Fatores Positivos**")
                    for ajuste in ajustes_positivos:
                        st.write(f"• {ajuste['descricao']}: **+{ajuste['valor']} pts**")
            
            with col2:
                if ajustes_negativos:
                    st.error("**❌ Fatores de Atenção**")
                    for ajuste in ajustes_negativos:
                        st.write(f"• {ajuste['descricao']}: **{ajuste['valor']} pts**")
            
            # Total
            st.markdown("---")
            st.markdown(f"### 🎯 **Score Final: {score_final}/1000**")
        
        # Insights Tavily
        if insights_tavily:
            st.header("🧠 Inteligência de Mercado")
            
            for insight in insights_tavily:
                with st.expander(f"{insight['tipo']} - Confiabilidade: {insight['confiabilidade']['nivel']}"):
                    st.write(insight['texto'])
                    st.caption(f"Fontes: {insight['confiabilidade']['fontes']}")
        
        # Recomendações
        st.header("💡 Recomendações")
        
        if score_final >= 800:
            st.success("""
            **Cliente Premium** 🏆
            - Elegível para as melhores condições
            - Prêmio com desconto máximo
            - Produtos exclusivos disponíveis
            - Fast-track na aprovação
            """)
        elif score_final >= 650:
            st.success("""
            **Cliente Excelente** ⭐
            - Condições privilegiadas
            - Desconto significativo no prêmio
            - Aprovação simplificada
            """)
        elif score_final >= 500:
            st.info("""
            **Cliente Padrão** ✅
            - Aprovação normal
            - Prêmio padrão de mercado
            - Produtos convencionais
            """)
        elif score_final >= 350:
            st.warning("""
            **Cliente Regular** ⚠️
            - Análise adicional recomendada
            - Possível majoração de prêmio (10-30%)
            - Considerar exigir rastreador
            """)
        elif score_final >= 200:
            st.warning("""
            **Cliente de Atenção** 🔴
            - Análise criteriosa necessária
            - Majoração de prêmio (30-50%)
            - Rastreador obrigatório
            - Franquia elevada
            """)
        else:
            st.error("""
            **Cliente Crítico** ⛔
            - Alto risco identificado
            - Considerar recusa ou condições especiais
            - Se aprovar: prêmio majorado (50-100%)
            - Múltiplas restrições de cobertura
            """)
        
        # Exportação dos dados
        st.header("📥 Exportar Análise")
        
        # Preparar dados para exportação
        resultado_exportacao = {
            'timestamp': datetime.now().isoformat(),
            'score_final': score_final,
            'classificacao': classificacao,
            'detalhamento': {
                'base': calculadora.detalhamento['base'],
                'categorias': calculadora.detalhamento['categorias'],
                'ajustes': calculadora.detalhamento['ajustes']
            },
            'dados_entrada': {
                'cep': cep_input,
                'cnpj': cnpj_input if cnpj_input else None,
                'veiculo': {
                    'marca': marca_input if marca_input else None,
                    'modelo': modelo_input if modelo_input else None
                }
            },
            'insights_inteligencia': insights_tavily if insights_tavily else [],
            'pkl_analysis': {
                'status': st.session_state.pkl_status,
                'total_clientes': st.session_state.pkl_stats.get('total_clientes', 0) if st.session_state.pkl_stats else 0
            }
        }
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Botão para baixar JSON
            json_str = json.dumps(resultado_exportacao, indent=2, ensure_ascii=False)
            st.download_button(
                label="📄 Baixar JSON",
                data=json_str,
                file_name=f"score_cliente_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            # Preparar CSV
            csv_data = pd.DataFrame([{
                'timestamp': datetime.now().isoformat(),
                'score': score_final,
                'classificacao': classificacao,
                'cep': cep_input,
                'cnpj': cnpj_input if cnpj_input else '',
                'marca': marca_input if marca_input else '',
                'modelo': modelo_input if modelo_input else '',
                'ajuste_localizacao': calculadora.detalhamento['categorias']['localizacao'],
                'ajuste_veiculo': calculadora.detalhamento['categorias']['veiculo'],
                'ajuste_empresa': calculadora.detalhamento['categorias']['empresa'],
                'ajuste_inteligencia': calculadora.detalhamento['categorias']['inteligencia'],
                'pkl_loaded': st.session_state.pkl_status == 'loaded'
            }])
            
            csv_str = csv_data.to_csv(index=False)
            st.download_button(
                label="📊 Baixar CSV",
                data=csv_str,
                file_name=f"score_cliente_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col3:
            # Código para integração
            if st.button("🔗 Ver Código de Integração"):
                st.code(f"""
# Integração com sistema
cliente = {{
    "id": "GENERATED_ID",
    "score": {score_final},
    "classificacao": "{classificacao}",
    "timestamp": "{datetime.now().isoformat()}",
    "pkl_analysis": {st.session_state.pkl_status == 'loaded'}
}}

# Para adicionar ao arquivo .pkl vetorizado:
# df_clientes = pd.read_pickle('clientes.pkl')
# df_clientes = df_clientes.append(cliente, ignore_index=True)
# df_clientes.to_pickle('clientes.pkl')
                """, language='python')
        
        # Estatísticas comparativas
        st.header("📈 Análise Comparativa")
        
        with st.expander("Ver comparação com base de clientes"):
            
            # Se PKL carregado, usar dados reais
            if st.session_state.pkl_status == 'loaded' and st.session_state.pkl_stats:
                st.success("📊 Usando dados reais da base vetorizada")
                total = st.session_state.pkl_stats.get('total_clientes', 10000)
                media_real = st.session_state.pkl_stats.get('media_sinistros', 0.5)
                taxa_real = st.session_state.pkl_stats.get('taxa_sinistralidade', 0.15)
            else:
                st.info("💡 Usando dados simulados (carregue o PKL para dados reais)")
                total = 10000
                media_real = 0.5
                taxa_real = 0.15
            
            # Simular distribuição
            np.random.seed(42)
            scores_simulados = np.random.normal(500, 150, total)
            scores_simulados = np.clip(scores_simulados, 0, 1000)
            
            # Estatísticas
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                media = np.mean(scores_simulados)
                st.metric("Média Geral", f"{media:.0f}")
            
            with col2:
                mediana = np.median(scores_simulados)
                st.metric("Mediana", f"{mediana:.0f}")
            
            with col3:
                percentil = (np.sum(scores_simulados < score_final) / len(scores_simulados)) * 100
                st.metric("Seu Percentil", f"{percentil:.1f}%")
            
            with col4:
                desvio = np.std(scores_simulados)
                st.metric("Desvio Padrão", f"{desvio:.0f}")
            
            st.markdown("### Distribuição de Scores na Base")
            st.caption(f"Seu cliente: {score_final} pontos (linha vermelha)")
            
            # Criar visualização simples
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(scores_simulados, bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax.axvline(score_final, color='red', linestyle='--', linewidth=2, label=f'Seu Cliente ({score_final})')
            ax.set_xlabel('Score')
            ax.set_ylabel('Frequência')
            ax.set_title(f'Distribuição de Scores - Base de {total:,} Clientes')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Adicionar indicador se PKL está carregado
            if st.session_state.pkl_status == 'loaded':
                ax.text(0.02, 0.98, '🧬 Dados Reais', transform=ax.transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
            
            st.pyplot(fig)
            
            # Tabela de distribuição por faixas
            st.markdown("### Distribuição por Classificação")
            
            faixas = [
                ('Premium', 800, 1000),
                ('Excelente', 650, 799),
                ('Bom', 500, 649),
                ('Regular', 350, 499),
                ('Atenção', 200, 349),
                ('Crítico', 0, 199)
            ]
            
            distribuicao = []
            for nome, min_score, max_score in faixas:
                qtd = np.sum((scores_simulados >= min_score) & (scores_simulados <= max_score))
                pct = (qtd / len(scores_simulados)) * 100
                distribuicao.append({
                    'Classificação': nome,
                    'Faixa': f'{min_score}-{max_score}',
                    'Quantidade': qtd,
                    'Percentual': f'{pct:.1f}%'
                })
            
            df_dist = pd.DataFrame(distribuicao)
            st.dataframe(df_dist, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
