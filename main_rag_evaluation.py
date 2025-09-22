# Sistema RAG Médico - Hospital Barros Luco, Santiago de Chile
# Versión: 3.0 - Adaptado para asistencia médica y gestión hospitalaria
import streamlit as st
import os
import json
import time
import uuid
from datetime import datetime
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory#Mantiene los mensajes recientes completos y resume conversaciones antiguas para ahorrar tokens
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Archivo: main_rag_evaluation.py
# Sistema RAG Médico con Evaluación Avanzada

# Librerías básicas
import streamlit as st
import time
import uuid
from datetime import datetime
from collections import Counter
import os
import openai
from openai import OpenAI

# Importaciones de LangChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import Document

# Librerías para análisis y visualización
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

# Cargar variables de entorno desde archivo .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    st.warning("⚠ python-dotenv no está instalado. Instálalo con: pip install python-dotenv")

# variables de entorno :)
github_token = os.getenv("GITHUB_TOKEN")
github_base_url = os.getenv("GITHUB_BASE_URL", "https://models.inference.ai.azure.com")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "hospital-barros-luco")

# Configuracion de LangSmith si está disponible
if LANGSMITH_TRACING.lower() == "true" and LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT

if github_token:
    os.environ["OPENAI_API_KEY"] = github_token
    os.environ["OPENAI_API_BASE"] = github_base_url
else:
    st.error("❌ GITHUB_TOKEN environment variable is not set. Please check your .env file.")
    st.info("💡 Make sure your .env file contains: GITHUB_TOKEN=your_token_here")
    st.stop()

st.set_page_config(page_title="🏥 Hospital Barros Luco - Asistente IA", page_icon="🏥", layout="wide")  # Configura página web del hospital

class ChatbotMedicoRAG:  # Clase principal del chatbot médico
    def __init__(self):
        self.client = None  # Cliente OpenAI para consultas médicas
        self.llm = None  # Modelo de lenguaje especializado en medicina
        self.embeddings_model = None  # Modelo para crear vectores de información médica
        self.vectorstore = None  # Base de datos vectorial de conocimiento médico
        self.conversation_chain = None  # Cadena conversacional para consultas médicas
        self.memory = None  # Sistema de memoria para historial de consultas
        self.documents = []  # Lista de documentos médicos y protocolos
        self.embeddings = None  # Vectores generados de documentos médicos
        self.interaction_logs = []  # Registro de todas las consultas médicas
        
    def initialize_client(self):
        """Inicializa el cliente de OpenAI"""
        if not github_token:
            st.error("❌ GitHub token not available")
            return False
        
        try:
            self.client = OpenAI(  # Crea cliente OpenAI directo
                base_url=github_base_url,
                api_key=github_token
            )
            return True
        except Exception as e:
            st.error(f"Error initializing client: {str(e)}")
            return False

    def initialize_llm(self):
        """Inicializa el modelo de lenguaje de LangChain"""
        try:
            self.llm = ChatOpenAI(  # Configura modelo GPT-4o-mini con temperatura baja
                model="gpt-4o-mini",
                temperature=0.1,  # Temperatura baja = respuestas más serio
                openai_api_key=github_token,
                openai_api_base=github_base_url
            )
            return True
        except Exception as e:
            st.error(f"Error initializing LLM: {str(e)}")
            return False

    def initialize_embeddings(self):
        """Inicializa el modelo de embeddings de LangChain"""
        if not github_token:
            st.error("❌ GitHub token not available for embeddings")
            return False
        
        try:
            self.embeddings_model = OpenAIEmbeddings(  # Modelo para convertir texto en vectores
                model="text-embedding-3-small",  # Modelo pequeño pero eficiente
                openai_api_key=github_token,
                openai_api_base=github_base_url
            )
            return True
        except Exception as e:
            st.error(f"Error initializing embeddings: {str(e)}")
            return False

    def setup_memory(self):
        """Configura la memoria conversacional"""
        try:
            self.memory = ConversationSummaryBufferMemory(  # Memoria híbrida médica
                llm=self.llm,
                max_token_limit=2000,  # Límite de tokens para evitar contexto muy largo
                memory_key="chat_history",  # Clave para acceder al historial
                return_messages=True,  # Devuelve mensajes estructurados
                output_key="answer"  # Clave para las respuestas del asistente
            )
            return True
        except Exception as e:
            st.error(f"Error setting up memory: {str(e)}")
            return False

    def get_embeddings_langchain(self, texts):
        """Obtiene embeddings usando LangChain"""
        try:
            if isinstance(texts[0], str):  # Si son strings, convertir a documentos
                documents = [Document(page_content=text) for text in texts]
            else:
                documents = texts
            
            embeddings = self.embeddings_model.embed_documents([doc.page_content for doc in documents])  # Convierte textos en vectores
            return np.array(embeddings)  # Devuelve como array numpy
        except Exception as e:
            st.error(f"Error getting embeddings: {str(e)}")
            return None

    def get_query_embedding_langchain(self, query):
        """Obtiene embedding de consulta usando LangChain"""
        try:
            embedding = self.embeddings_model.embed_query(query)  # Convierte pregunta en vector
            return np.array(embedding)
        except Exception as e:
            st.error(f"Error getting query embedding: {str(e)}")
            return None

    def hybrid_search_with_metrics(self, query, top_k=5):
        """Realiza búsqueda híbrida con métricas detalladas"""
        if not self.embeddings_model or not self.documents or self.embeddings is None:
            return [], 0.0
            
        start_time = time.time()  # Inicia cronómetro para métricas
        
        # Get query embedding
        query_embedding = self.get_query_embedding_langchain(query)  # Convierte pregunta a vector
        if query_embedding is None:
            return [], 0.0
        
        # Semantic similarity
        semantic_similarities = cosine_similarity([query_embedding], self.embeddings)[0]  # Calcula similitud semántica (70%)
        
        # Keyword similarity
        keyword_scores = []  # Lista para almacenar puntuaciones de palabras clave
        query_words = set(query.lower().split())  # Convierte pregunta en conjunto de palabras
        for doc in self.documents:
            doc_words = set(doc.lower().split())  # Convierte documento en conjunto de palabras
            overlap = len(query_words.intersection(doc_words))  # Cuenta palabras en común
            keyword_scores.append(overlap / max(len(query_words), 1))  # Calcula porcentaje de coincidencia
        
        # Combined score
        combined_scores = 0.7 * semantic_similarities + 0.3 * np.array(keyword_scores)  # 70% semántica + 30% keywords
        top_indices = np.argsort(combined_scores)[::-1][:top_k]  # Obtiene índices de mejores resultados
        
        results = []
        for idx in top_indices:
            results.append({  # Crea diccionario con toda la información del resultado
                'document': self.documents[idx],
                'semantic_score': semantic_similarities[idx],
                'keyword_score': keyword_scores[idx],
                'combined_score': combined_scores[idx],
                'index': idx
            })
        
        retrieval_time = time.time() - start_time  # Calcula tiempo total de búsqueda
        return results, retrieval_time

    def generate_response_with_metrics(self, query, context_docs):
        """Genera respuesta con métricas de tiempo y filtrado adecuado de fuentes"""
        if not self.client:
            return "Error: Cliente no disponible", 0.0
            
        start_time = time.time()  # Inicia cronómetro
        
        # Filtra documentos de contexto relevantes
        relevant_docs = []  # Lista para documentos realmente relevantes
        query_words = set(query.lower().split())  # Palabras de la pregunta
        stop_words = {'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 'como', 'qué', 'cómo', 'cuál', 'hola', 'gracias'}  # Palabras sin significado
        meaningful_words = query_words - stop_words  # Solo palabras importantes
        
        for doc in context_docs:
            content = doc['document'].lower()
            # Verifica si el documento es realmente relevante
            if len(meaningful_words) > 0:  # Si hay palabras importantes
                content_words = set(content.split())
                if meaningful_words.intersection(content_words):  # Si encuentra palabras importantes en el documento
                    relevant_docs.append(doc)  # Agrega documento a la lista relevante
            elif len(query_words) <= 2:  # Very short queries like "hola"
                continue  # Don't use any context for greetings
        
        # Decide whether to use context or general knowledge
        if relevant_docs:  # Si hay documentos relevantes
            context = "\n".join([f"Documento {i+1}: {doc['document']}" 
                                for i, doc in enumerate(relevant_docs)])  # Crea contexto con documentos relevantes
            
            prompt = f"""Eres el asistente virtual del Hospital Barros Luco en Santiago de Chile. 

Contexto médico e información del hospital:
{context}

Consulta del paciente o visitante: {query}

Proporciona una respuesta clara, empática y basada en los protocolos del hospital. 
Si es una emergencia médica, recuerda al usuario que debe dirigirse inmediatamente al área de Urgencias."""  # Prompt médico hospitalario
        else:
            # No relevant context found
            prompt = f"""Eres el asistente virtual del Hospital Barros Luco en Santiago de Chile.

Consulta: {query}

🏥 Basándome en conocimiento médico general (esta información no está en nuestros protocolos específicos): 

Proporciona información útil pero siempre recuerda al usuario que debe consultar con personal médico del hospital para diagnósticos o tratamientos específicos."""  # Prompt médico general

        try:
            # Crea solicitud de completado a OpenAI
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",    # Modelo de OpenAI a usar
                messages=[{"role": "user", "content": prompt}],  # Mensaje para la IA
                temperature=0.1,        # Creatividad baja para respuestas precisas
                max_tokens=600          # Límite de palabras en respuesta
            )
            
            generation_time = time.time() - start_time  # Calcula tiempo total de generación
            response_text = response.choices[0].message.content  # Extrae texto de respuesta
            
            # Add prefix if using general knowledge
            if not relevant_docs:  # Si no usó base de datos
                response_text = f"🧠 Basándome en conocimiento general (no encontré información específica en la base de datos): {response_text}"  # Agrega prefijo indicativo
            
            return response_text, generation_time  # Retorna respuesta y tiempo
        except Exception as e:
            return f"Error generating response: {str(e)}", time.time() - start_time  # Retorna error si algo falla

    def evaluate_faithfulness(self, query, context, response):
        """Evalúa la fidelidad de la respuesta al contexto"""
        if not self.client:  # Verifica que el cliente esté disponible
            return 5.0
            
        # Crea prompt de evaluación para fidelidad
        eval_prompt = f"""Evalúa si la respuesta es fiel al contexto proporcionado.

Consulta: {query}

Contexto:
{context}

Respuesta:
{response}

¿La respuesta está basada únicamente en la información del contexto? 
Responde con un número del 1-10 donde:
- 1-3: Respuesta contradice o no está basada en el contexto
- 4-6: Respuesta parcialmente basada en el contexto
- 7-10: Respuesta completamente fiel al contexto

Responde SOLO con el número:"""  # Prompt específico para evaluar fidelidad

        try:
            # Genera evaluación usando OpenAI
            result = self.client.chat.completions.create(
                model="gpt-4o-mini",      # Modelo para evaluación
                messages=[{"role": "user", "content": eval_prompt}],  # Prompt de evaluación
                temperature=0.1,          # Baja creatividad para evaluación consistente
                max_tokens=10             # Solo necesita un número
            )
            return float(result.choices[0].message.content.strip())  # Convierte respuesta a número
        except:
            return 5.0  # Valor por defecto si hay error

    def evaluate_relevance(self, query, response):
        """Evalúa la relevancia de la respuesta a la consulta"""
        if not self.client:  # Verifica que el cliente esté disponible
            return 5.0
            
        # Crea prompt de evaluación para relevancia
        eval_prompt = f"""Evalúa qué tan relevante es la respuesta para la consulta.

Consulta: {query}

Respuesta: {response}

¿Qué tan bien responde la respuesta a la consulta?
Responde con un número del 1-10 donde:
- 1-3: Respuesta no relacionada o irrelevante
- 4-6: Respuesta parcialmente relevante
- 7-10: Respuesta muy relevante y útil

Responde SOLO con el número:"""  # Prompt específico para evaluar relevancia

        try:
            # Genera evaluación usando OpenAI
            result = self.client.chat.completions.create(
                model="gpt-4o-mini",      # Modelo para evaluación
                messages=[{"role": "user", "content": eval_prompt}],  # Prompt de evaluación
                temperature=0.1,          # Baja creatividad para consistencia
                max_tokens=10             # Solo necesita un número
            )
            return float(result.choices[0].message.content.strip())  # Convierte respuesta a número
        except:
            return 5.0  # Valor por defecto si hay error

    def evaluate_context_precision(self, query, retrieved_docs):
        """Evalúa la precisión del contexto"""
        if not self.client or not retrieved_docs:  # Verifica cliente y documentos
            return 0.0
        
        relevant_count = 0  # Contador de documentos relevantes
        for doc in retrieved_docs:  # Evalúa cada documento recuperado
            # Crea prompt de evaluación para cada documento
            eval_prompt = f"""¿Este documento es relevante para responder la consulta?

Consulta: {query}

Documento: {doc['document'][:300]}...

Responde SOLO 'SI' o 'NO':"""  # Prompt simple para evaluar relevancia del documento
            
            try:
                # Evaluate document relevance
                result = self.client.chat.completions.create(
                    model="gpt-4o-mini",      # Modelo para evaluación
                    messages=[{"role": "user", "content": eval_prompt}],  # Prompt de evaluación
                    temperature=0.1,          # Baja creatividad para respuestas consistentes
                    max_tokens=5              # Solo necesita SI/NO
                )
                if result.choices[0].message.content.strip().upper() == 'SI':  # Si el documento es relevante
                    relevant_count += 1  # Incrementa contador
            except:
                pass  # Ignora errores individuales
        
        return relevant_count / len(retrieved_docs)  # Retorna precisión (% de docs relevantes)

    def log_interaction(self, query, response, metrics, context_docs):
        """Registra la interacción para análisis"""
        # Crea entrada de registro con todos los datos de interacción
        log_entry = {
            'id': str(uuid.uuid4()),  # ID único para esta interacción
            'timestamp': datetime.now().isoformat(),  # Marca de tiempo actual
            'query': query,                           # Pregunta del usuario
            'response': response,                     # Respuesta generada
            'metrics': metrics,                       # Métricas de calidad
            'context_count': len(context_docs),       # Número de documentos usados
            'context_scores': [doc.get('combined_score', 0) for doc in context_docs]  # Puntuaciones de relevancia
        }
        
        self.interaction_logs.append(log_entry)  # Agrega al historial de interacciones

    def clean_placeholder_documents(self):
        """Elimina documentos de prueba y placeholder del almacén vectorial"""
        if not self.vector_store:  # Verifica que existe el vector store
            return False
            
        try:
            # Get all documents from ChromaDB
            collection = self.vector_store._collection  # Accede a la colección de ChromaDB
            all_docs = collection.get()                 # Obtiene todos los documentos
            
            if not all_docs['documents']:  # Si no hay documentos
                return True
                
            # Find indices of placeholder documents to remove
            indices_to_remove = []  # Lista de índices a eliminar
            for i, doc in enumerate(all_docs['documents']):  # Revisa cada documento
                if (doc and ('placeholder' in doc.lower() or   # Si contiene "placeholder"
                           'test' in doc.lower() or             # o "test" 
                           len(doc.strip()) < 10)):             # o es muy corto
                    indices_to_remove.append(i)  # Marca para eliminar
            
            # Remove placeholder documents if any found
            if indices_to_remove:  # Si hay documentos para eliminar
                ids_to_remove = [all_docs['ids'][i] for i in indices_to_remove]  # Obtiene IDs a eliminar
                collection.delete(ids=ids_to_remove)  # Elimina de ChromaDB
                print(f"Removed {len(ids_to_remove)} placeholder documents")  # Confirma eliminación
                return True
            
            return True  # Operación exitosa
        except Exception as e:
            print(f"Error cleaning placeholder documents: {e}")  # Muestra error si algo falla
            return False

def initialize_hospital_documents():
    """Inicializa con documentos médicos y protocolos del Hospital Barros Luco"""
    # Retorna lista de protocolos médicos, información de servicios y procedimientos
    return [
        # SERVICIOS Y DEPARTAMENTOS
        "El Hospital Barros Luco ubicado en Santiago de Chile cuenta con los siguientes servicios: Emergencias 24/7, Cuidados Intensivos (UCI), Cardiología, Neurología, Pediatría, Ginecología y Obstetricia, Oncología, Ortopedia y Traumatología, Radiología e Imágenes, Laboratorio Clínico, Farmacia, y Rehabilitación Física.",
        
        # HORARIOS Y CONTACTO
        "Horarios de atención: Emergencias 24 horas. Consultas externas: Lunes a Viernes 7:00 AM - 6:00 PM, Sábados 8:00 AM - 2:00 PM. Teléfono principal: (01) 234-5678. Emergencias: 911. Dirección: Av. Salud 123, Lima, Perú.",
        
        # PROTOCOLOS DE EMERGENCIA
        "Protocolo de emergencias: En caso de emergencia médica, dirigirse inmediatamente al área de Emergencias en el primer piso. El personal de triaje evaluará la urgencia. Código Azul: Paro cardiorrespiratorio. Código Rojo: Emergencia médica. Código Amarillo: Emergencia quirúrgica.",
        
        # PROCEDIMIENTOS DE HOSPITALIZACIÓN
        "Proceso de hospitalización: 1) Admisión con documento de identidad y seguro médico. 2) Evaluación médica inicial. 3) Asignación de habitación según disponibilidad. 4) Entrega de brazalete de identificación. 5) Orientación sobre normas hospitalarias. Horarios de visita: 2:00 PM - 4:00 PM y 6:00 PM - 8:00 PM.",
        
        # ESPECIALIDADES MÉDICAS
        "Cardiología: Diagnóstico y tratamiento de enfermedades del corazón. Servicios: Electrocardiograma, Ecocardiograma, Pruebas de esfuerzo, Cateterismo cardíaco, Marcapasos. Neurología: Atención de enfermedades del sistema nervioso. Servicios: Electroencefalograma, Resonancia magnética cerebral, Tratamiento de epilepsia, ACV, migrañas.",
        
        # ANÁLISIS Y LABORATORIO
        "Laboratorio Clínico: Análisis de sangre, orina, heces. Horario: 6:00 AM - 4:00 PM. Ayuno requerido: 12 horas para glucosa y perfil lipídico, 8 horas para química sanguínea. Resultados: Análisis básicos en 2-4 horas, cultivos en 48-72 horas, biopsias en 5-7 días laborables.",
        
        # IMAGENOLOGÍA
        "Servicios de Imagenología: Rayos X, Tomografía (TAC), Resonancia Magnética (RM), Ecografía, Mamografía. Preparación especial requerida para TAC con contraste y RM. Citas programadas de Lunes a Viernes. Urgencias atendidas las 24 horas en Emergencias.",
        
        # FARMACIA Y MEDICAMENTOS
        "Farmacia Hospitalaria: Dispensación de medicamentos para pacientes hospitalizados y externos. Horario: 24 horas para hospitalización, 7:00 AM - 8:00 PM para externos. Aceptamos recetas de médicos colegiados. Medicamentos controlados requieren receta especial y documento de identidad.",
        
        # PEDIATRÍA
        "Servicio de Pediatría: Atención médica para niños de 0 a 14 años. Consultas programadas y emergencias pediátricas. Vacunación según esquema nacional. Sala de juegos disponible. Acompañante permanente permitido. Horarios de visita especiales para familiares.",
        
        # MATERNIDAD
        "Servicio de Maternidad: Control prenatal, parto y puerperio. Sala de partos con tecnología avanzada. Programa de preparación para el parto. Lactancia materna promovida. Habitaciones individuales y compartidas disponibles. Visitas del padre permitidas las 24 horas.",
        
        # POLÍTICAS DEL HOSPITAL
        "Políticas hospitalarias: Prohibido fumar en todas las instalaciones. Uso obligatorio de mascarilla. Máximo 2 visitantes por paciente. Silencio en áreas de hospitalización. Dispositivos móviles en modo silencioso. Respeto al personal médico y de enfermería.",
        
        # SEGUROS Y PAGOS
        "Convenios de seguros: EsSalud, SIS, Pacífico Seguros, Rímac Seguros, La Positiva. Formas de pago: Efectivo, tarjetas de crédito/débito, transferencias bancarias. Facturación electrónica disponible. Planes de financiamiento para cirugías mayores.",
        
        # TECNOLOGÍA E INTELIGENCIA ARTIFICIAL EN SALUD
        "El uso de IA en radiología ayuda a detectar anomalías en imágenes médicas con mayor precisión. Nuestro Hospital Barros Luco utiliza sistemas de inteligencia artificial para análisis de radiografías, tomografías y resonancias magnéticas, lo que permite diagnósticos más rápidos y exactos.",
        
        "La telemedicina permite realizar consultas médicas a distancia, mejorando el acceso en zonas rurales. El Hospital Barros Luco ofrece servicios de teleconsulta para seguimiento de pacientes crónicos, consultas de especialidades y orientación médica inicial.",
        
        "Sistemas de IA para análisis de laboratorio: Algoritmos que ayudan en la interpretación de resultados de exámenes de sangre, orina y otros fluidos corporales, reduciendo errores humanos y acelerando el proceso diagnóstico.",
        
        "Asistentes virtuales médicos: Chatbots especializados como este sistema que brindan información hospitalaria 24/7, orientan sobre procedimientos y servicios, y facilitan el acceso a información médica básica para pacientes y familiares."
    ]

def create_medical_evaluation_dataset():
    # Crea dataset de evaluación sistemática para probar el rendimiento del sistema RAG médico
    # Se usa en la pestaña "🧪 Evaluación Sistemática" para ejecutar pruebas automáticas de consultas médicas
    # Mide métricas de calidad (fidelidad, relevancia, precisión) y genera reportes de rendimiento médico
    # Cada elemento contiene: query (consulta médica), expected_context (tipo de info médica), ground_truth (respuesta médica esperada)
    return [
        {
            "query": "¿Cuáles son los horarios de atención del hospital?",
            "expected_context": "horarios de servicios hospitalarios",
            "ground_truth": "Emergencias 24 horas. Consultas externas: Lunes a Viernes 7:00 AM - 6:00 PM, Sábados 8:00 AM - 2:00 PM."
        },
        {
            "query": "¿Qué servicios tiene el Hospital Barros Luco?",
            "expected_context": "listado de servicios médicos",
            "ground_truth": "El hospital cuenta con Emergencias, UCI, Cardiología, Neurología, Pediatría, Ginecología, Oncología, Ortopedia, Radiología, Laboratorio y Farmacia."
        },
        {
            "query": "¿Cómo es el proceso de hospitalización?",
            "expected_context": "procedimientos de ingreso hospitalario",
            "ground_truth": "Requiere admisión con documento y seguro, evaluación médica, asignación de habitación, brazalete de identificación y orientación sobre normas."
        },
        {
            "query": "¿Qué estudios se realizan en el laboratorio?",
            "expected_context": "servicios de laboratorio clínico",
            "ground_truth": "Se realizan análisis de sangre, orina, heces. Horario 6:00 AM - 4:00 PM. Resultados básicos en 2-4 horas."
        },
        {
            "query": "¿Cuáles son los horarios de visita?",
            "expected_context": "políticas de visitas hospitalarias",
            "ground_truth": "Horarios de visita: 2:00 PM - 4:00 PM y 6:00 PM - 8:00 PM. Máximo 2 visitantes por paciente."
        },
        {
            "query": "¿Cómo ayuda la IA en radiología?",
            "expected_context": "tecnología e inteligencia artificial en salud",
            "ground_truth": "La IA en radiología ayuda a detectar anomalías en imágenes médicas con mayor precisión, permitiendo diagnósticos más rápidos y exactos."
        },
        {
            "query": "¿Qué es la telemedicina y cómo funciona?",
            "expected_context": "servicios de telemedicina",
            "ground_truth": "La telemedicina permite realizar consultas médicas a distancia, mejorando el acceso en zonas rurales. Incluye teleconsulta, seguimiento de pacientes crónicos y orientación médica."
        },
        {
            "query": "¿Qué tecnologías de IA usa el hospital?",
            "expected_context": "sistemas de inteligencia artificial hospitalarios",
            "ground_truth": "El hospital usa IA para análisis radiológico, interpretación de laboratorio, asistentes virtuales médicos y sistemas de diagnóstico automatizado."
        },
    ]

def export_langsmith_format(logs):
    """Exporta logs en formato LangSmith"""
    langsmith_data = []
    for log in logs:
        langsmith_data.append({
            "run_id": log['id'],
            "timestamp": log['timestamp'],
            "inputs": {"query": log['query']},
            "outputs": {"response": log['response']},
            "metrics": log['metrics'],
            "metadata": {
                "context_count": log['context_count'],
                "context_scores": log['context_scores']
            }
        })
    return langsmith_data

def main():
    # pagina streamlit 
    st.set_page_config(
        page_title="RAG Chatbot",         # Título de la página
        page_icon="🤖",                   # Icono del navegador
        layout="wide",                    # Diseño ancho para aprovechar pantalla
        initial_sidebar_state="expanded"  # Barra lateral expandida
    )
    
    # mero css
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);  /* Gradiente de color */
        padding: 1rem;               /* Espaciado interno */
        border-radius: 10px;         /* Bordes redondeados */
        color: white;                /* Texto blanco */
        text-align: center;          /* Centrado */
        margin-bottom: 2rem;         /* Margen inferior */
    }
    .metric-card {
        background: #f0f2f6;         /* Fondo gris claro */
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;  /* Borde izquierdo de color */
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background: #e3f2fd;         /* Fondo azul para usuario */
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background: #f3e5f5;         /* Fondo rosa para asistente */
        border-left: 4px solid #9c27b0;
    }
    .source-card {
        background: #fff3e0;         /* Fondo naranja claro para fuentes */
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 3px solid #ff9800;  /* Borde naranja */
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Hospital Barros Luco - Santiago de Chile - Asistente Virtual</h1>
        <p>Sistema de consultas médicas inteligente con información actualizada del hospital</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Verifica si el token de GitHub está disponible
    if not github_token:  # Verifica si existe el token de GitHub
        st.error("❌ Please check your .env file and make sure GITHUB_TOKEN is set.")
        st.info("💡 Your .env file should contain: GITHUB_TOKEN=your_token_here")
        return

    # Inicializa el estado de la sesión para datos persistentes
    if "chatbot_rag" not in st.session_state:  # Si es la primera vez
        st.session_state.chatbot_rag = ChatbotMedicoRAG()  # Crea instancia del chatbot médico
        st.session_state.chatbot_rag.documents = initialize_hospital_documents()  # Carga documentos médicos del hospital

        # Inicializa todos los componentes del sistema de forma secuencial
        if st.session_state.chatbot_rag.initialize_client():        # Inicializa cliente OpenAI
            if st.session_state.chatbot_rag.initialize_llm():       # Configura modelo de lenguaje
                if st.session_state.chatbot_rag.initialize_embeddings():  # Inicializa embeddings
                    st.session_state.chatbot_rag.setup_memory()     # Configura memoria conversacional
    
    # Estado de mensajes del chat
    if 'messages' not in st.session_state:  # Si es la primera sesión
        st.session_state.messages = []      # Lista vacía para mensajes del chat

    # Panel lateral de control (la sidebar)
    with st.sidebar:
        st.markdown("### 🎛 Panel de Control")  # Título del panel lateral
        
        # Estado del sistema
        st.markdown("#### 📊 Estado del Sistema")
        if st.session_state.chatbot_rag.embeddings is not None:  # Si los embeddings están listos
            st.success("✅ Embeddings Activos")
        else:
            st.warning("⚠ Embeddings Pendientes")  # Si faltan embeddings
            
        if st.session_state.chatbot_rag.documents:  # Si hay documentos cargados
            st.info(f"📚 {len(st.session_state.chatbot_rag.documents)} Documentos Cargados")
        else:
            st.error("❌ Sin Documentos")  # Si no hay documentos
        
        # Configuraciones
        st.markdown("#### ⚙ Configuraciones")
        temperature = st.slider("🌡 Temperatura", 0.0, 1.0, 0.1, 0.1)
        max_tokens = st.slider("📝 Max Tokens", 100, 1000, 600, 50)
        top_k = st.slider("🔍 Top K Docs", 1, 10, 3, 1)
        
    # Crea pestañas con diseño mejorado
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏥 Consultas Médicas", 
        "  Protocolos y Documentos", 
        "📊 Métricas del Sistema", 
        "🧪 Evaluación de Calidad", 
        "📈 Reportes Hospitalarios"
    ])
    
    with tab1:
        st.markdown("### 🏥 Asistente Virtual del Hospital Barros Luco")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Chat interface con diseño mejorado
            chat_container = st.container()
            with chat_container:
                for message in st.session_state.messages:
                    if message["role"] == "user":
                        st.markdown(f"""
                        <div class="chat-message user-message">
                            <strong>👤 Tu:</strong><br>
                            {message["content"]}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="chat-message assistant-message">
                            <strong>🤖 Asistente:</strong><br>
                            {message["content"]}
                        </div>
                        """, unsafe_allow_html=True)
            
            # Chat input con placeholder mejorado
            if prompt := st.chat_input("🏥 Pregúntame sobre horarios, servicios médicos, procedimientos..."):
                # Añade mensaje del usuario
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Genera respuesta
                with st.chat_message("assistant"):
                    with st.spinner("Procesando con métricas..."):
                        chatbot = st.session_state.chatbot_rag
                        
                        if chatbot.embeddings is None:
                            st.warning("Genera embeddings primero en la pestaña de documentos")
                        else:
                            # Búsqueda híbrida con métricas
                            results, retrieval_time = chatbot.hybrid_search_with_metrics(prompt, top_k=3)
                            
                            if results:
                                # Genera respuesta
                                response, generation_time = chatbot.generate_response_with_metrics(prompt, results)
                                
                                # Calcula métricas
                                metrics = {
                                    'retrieval_time': retrieval_time,
                                    'generation_time': generation_time,
                                    'total_time': retrieval_time + generation_time,
                                    'docs_retrieved': len(results),
                                    'avg_relevance_score': np.mean([r['combined_score'] for r in results])
                                }
                                
                                # Evalúa fidelidad, relevancia y precisión del contexto
                                context_text = "\n".join([r['document'] for r in results])
                                metrics['faithfulness'] = chatbot.evaluate_faithfulness(prompt, context_text, response)
                                metrics['relevance'] = chatbot.evaluate_relevance(prompt, response)
                                metrics['context_precision'] = chatbot.evaluate_context_precision(prompt, results)
                                
                                # Muestra la respuesta
                                st.markdown(response)
                                
                                # Muestra las métricas
                                with st.expander("📊 Métricas de esta respuesta"):
                                    # Filtra fuentes relevantes para métricas
                                    query_words = set(prompt.lower().split())
                                    stop_words = {'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 'como', 'qué', 'cómo', 'cuál', 'hola', 'gracias'}
                                    meaningful_words = query_words - stop_words
                                    
                                    relevant_count = 0
                                    for result in results:
                                        content = result['document'].lower()
                                        if 'placeholder' in content or 'test' in content or len(content.strip()) < 10:
                                            continue
                                        if len(meaningful_words) > 0:
                                            content_words = set(content.split())
                                            if meaningful_words.intersection(content_words):
                                                relevant_count += 1
                                    
                                    col_a, col_b, col_c, col_d = st.columns(4)
                                    with col_a:
                                        st.markdown(f"""
                                        <div class="metric-card">
                                            <h4>⏱ Tiempo Total</h4>
                                            <h2>{metrics['total_time']:.2f}s</h2>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    with col_b:
                                        fidelity_color = "green" if metrics['faithfulness'] >= 8 else "orange" if metrics['faithfulness'] >= 6 else "red"
                                        st.markdown(f"""
                                        <div class="metric-card">
                                            <h4>🎯 Fidelidad</h4>
                                            <h2 style="color: {fidelity_color}">{metrics['faithfulness']:.1f}/10</h2>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    with col_c:
                                        relevance_color = "green" if metrics['relevance'] >= 8 else "orange" if metrics['relevance'] >= 6 else "red"
                                        st.markdown(f"""
                                        <div class="metric-card">
                                            <h4>🔍 Relevancia</h4>
                                            <h2 style="color: {relevance_color}">{metrics['relevance']:.1f}/10</h2>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    with col_d:
                                        st.markdown(f"""
                                        <div class="metric-card">
                                            <h4>📚 Docs Relevantes</h4>
                                            <h2>{relevant_count}/{metrics['docs_retrieved']}</h2>
                                        </div>
                                        """, unsafe_allow_html=True)
                                
                                # Muestra las fuentes con filtrado
                                with st.expander(f"📚 Fuentes utilizadas"):
                                    # Filtra documentos placeholder e irrelevantes
                                    query_words = set(prompt.lower().split())
                                    stop_words = {'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 'como', 'qué', 'cómo', 'cuál', 'hola', 'gracias'}
                                    meaningful_words = query_words - stop_words
                                    
                                    relevant_sources = []
                                    for result in results:
                                        content = result['document'].lower()
                                        # Omite documentos placeholder
                                        if 'placeholder' in content or 'test' in content or len(content.strip()) < 10:
                                            continue
                                        # Verifica relevancia para consultas significativas
                                        if len(meaningful_words) > 0:
                                            content_words = set(content.split())
                                            if meaningful_words.intersection(content_words):
                                                relevant_sources.append(result)
                                    
                                    if relevant_sources:
                                        st.write(f"*Documentos analizados:* {len(results)} | *Fuentes relevantes:* {len(relevant_sources)}")
                                        for i, result in enumerate(relevant_sources[:3]):  # Show only first 3 relevant sources
                                            score_color = "green" if result['combined_score'] > 0.7 else "orange" if result['combined_score'] > 0.5 else "red"
                                            st.markdown(f"""
                                            <div class="source-card">
                                                <h4>📄 Fuente {i+1} 
                                                    <span style="color: {score_color}; font-size: 0.8em;">
                                                        (Score: {result['combined_score']:.3f})
                                                    </span>
                                                </h4>
                                                <p>{result['document'][:200]}...</p>
                                            </div>
                                            """, unsafe_allow_html=True)
                                            st.divider()
                                    else:
                                        st.info("🧠 Esta respuesta se basa en conocimiento general, no en documentos específicos de la base de datos.")
                                
                                # Registra la interacción
                                chatbot.log_interaction(prompt, response, metrics, results)
                                
                                # Añade respuesta a la sesión
                                st.session_state.messages.append({"role": "assistant", "content": response})
                            else:
                                st.error("Error en la búsqueda de documentos")
        
        with col2:
            st.markdown("### 🎛 Centro de Control")
            
            # Estado del sistema con indicadores
            st.markdown("#### 📊 Estado del Sistema")
            if st.session_state.chatbot_rag.embeddings is not None:
                st.success("✅ Embeddings Activos")
            else:
                st.warning("⚠ Embeddings Pendientes")
                
            if st.session_state.chatbot_rag.documents:
                st.info(f"📚 {len(st.session_state.chatbot_rag.documents)} Documentos")
            else:
                st.error("❌ Sin Documentos")
            
            st.markdown("#### 🔧 Acciones Rápidas")
            
            if st.button("🗑 Limpiar Chat", use_container_width=True):
                st.session_state.messages = []
                if st.session_state.chatbot_rag.memory:
                    st.session_state.chatbot_rag.memory.clear()
                st.rerun()
            
            if st.button("🔄 Generar Embeddings"):
                if st.session_state.chatbot_rag.documents and st.session_state.chatbot_rag.embeddings_model:
                    with st.spinner("Generando embeddings..."):
                        embeddings = st.session_state.chatbot_rag.get_embeddings_langchain(
                            st.session_state.chatbot_rag.documents
                        )
                        if embeddings is not None:
                            st.session_state.chatbot_rag.embeddings = embeddings
                            st.success("✅ Embeddings listos")
                        else:
                            st.error("❌ Error generando embeddings")
                else:
                    st.warning("Documentos no disponibles")
            
            
            # Estadísticas rápidas
            st.markdown("####   Estadísticas Rápidas")
            if st.session_state.chatbot_rag.interaction_logs:
                total_interactions = len(st.session_state.chatbot_rag.interaction_logs)
                avg_time = np.mean([log['metrics']['total_time'] for log in st.session_state.chatbot_rag.interaction_logs])
                st.metric("  Conversaciones", total_interactions)
                st.metric("⏱ Tiempo Promedio", f"{avg_time:.2f}s")
            else:
                st.info("📊 Sin datos aún")
    
    with tab2:
        st.header("📄 Gestión de Documentos")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📚 Documentos Actuales")
            
            for i, doc in enumerate(st.session_state.chatbot_rag.documents):
                with st.expander(f"Documento {i+1} ({len(doc)} caracteres)"):
                    st.text_area(
                        f"Contenido:",
                        value=doc,
                        height=100,
                        key=f"doc_display_{i}",
                        disabled=True
                    )
                    
                    col_edit, col_delete = st.columns(2)
                    with col_edit:
                        if st.button(f"✏ Editar", key=f"edit_{i}"):
                            st.session_state[f'editing_doc_{i}'] = True
                    
                    with col_delete:
                        if st.button(f"🗑 Eliminar", key=f"delete_{i}"):
                            st.session_state.chatbot_rag.documents.pop(i)
                            st.session_state.chatbot_rag.embeddings = None
                            st.rerun()
                    
                    # Edición online del contenido
                    if st.session_state.get(f'editing_doc_{i}', False):
                        new_content = st.text_area(
                            "Editar contenido:",
                            value=doc,
                            height=150,
                            key=f"edit_content_{i}"
                        )
                        
                        col_save, col_cancel = st.columns(2)
                        with col_save:
                            if st.button(f"💾 Guardar", key=f"save_{i}"):
                                st.session_state.chatbot_rag.documents[i] = new_content
                                st.session_state[f'editing_doc_{i}'] = False
                                st.session_state.chatbot_rag.embeddings = None
                                st.success("Documento actualizado")
                                st.rerun()
                        
                        with col_cancel:
                            if st.button(f"❌ Cancelar", key=f"cancel_{i}"):
                                st.session_state[f'editing_doc_{i}'] = False
                                st.rerun()
        
        with col2:
            st.subheader("➕ Agregar Documento")
            
            new_doc = st.text_area(
                "Contenido del nuevo documento:",
                height=200,
                placeholder="Escribe aquí el contenido del nuevo documento..."
            )
            
            if st.button("📝 Agregar Documento"):
                if new_doc.strip():
                    st.session_state.chatbot_rag.documents.append(new_doc.strip())
                    st.session_state.chatbot_rag.embeddings = None
                    st.success("Documento agregado exitosamente")
                    st.rerun()
                else:
                    st.warning("El documento no puede estar vacío")
            
            st.subheader("📊 Estadísticas")
            st.metric("Total documentos", len(st.session_state.chatbot_rag.documents))
            
            if st.session_state.chatbot_rag.documents:
                avg_length = np.mean([len(doc) for doc in st.session_state.chatbot_rag.documents])
                st.metric("Longitud promedio", f"{avg_length:.0f} caracteres")
                
                total_words = sum([len(doc.split()) for doc in st.session_state.chatbot_rag.documents])
                st.metric("Total palabras", f"{total_words:,}")
    
    with tab3:
        st.header("📊 Dashboard de Métricas")
        
        if st.session_state.chatbot_rag.interaction_logs:
            df = pd.DataFrame([
                {
                    'timestamp': log['timestamp'],
                    'query_length': len(log['query']),
                    'response_length': len(log['response']),
                    **log['metrics']
                }
                for log in st.session_state.chatbot_rag.interaction_logs
            ])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("⏱ Tiempo de Respuesta")
                fig = px.line(df, x='timestamp', y='total_time', title="Tiempo Total por Consulta")
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("📊 Distribución de Fidelidad")
                fig = px.histogram(df, x='faithfulness', title="Distribución de Puntuaciones de Fidelidad")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🔍 Métricas de Recuperación")
                fig = px.scatter(df, x='retrieval_time', y='generation_time', 
                               size='docs_retrieved', title="Tiempo de Recuperación vs Generación")
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("🎯 Calidad vs Precisión")
                fig = px.scatter(df, x='context_precision', y='relevance', 
                               title="Precisión de Contexto vs Relevancia")
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📈 Estadísticas Generales")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Consultas", len(df))
            with col2:
                st.metric("Tiempo Promedio", f"{df['total_time'].mean():.2f}s")
            with col3:
                st.metric("Fidelidad Promedio", f"{df['faithfulness'].mean():.1f}/10")
            with col4:
                st.metric("Relevancia Promedio", f"{df['relevance'].mean():.1f}/10")
        else:
            st.info("No hay datos de interacciones aún. Realiza algunas consultas primero.")
    
    with tab4:
        st.header("🧪 Evaluación Sistemática")
        
        if st.button("🧪 Ejecutar Evaluación Completa"):
            if st.session_state.chatbot_rag.embeddings is None:
                st.warning("Genera embeddings primero")
            else:
                eval_dataset = create_medical_evaluation_dataset()
                results = []
                
                with st.spinner("Ejecutando evaluación sistemática..."):
                    for test_case in eval_dataset:
                        query = test_case['query']
                        
                        docs, retrieval_time = st.session_state.chatbot_rag.hybrid_search_with_metrics(query, 3)
                        
                        if docs:
                            response, generation_time = st.session_state.chatbot_rag.generate_response_with_metrics(query, docs)
                            
                            context_text = "\n".join([d['document'] for d in docs])
                            faithfulness = st.session_state.chatbot_rag.evaluate_faithfulness(query, context_text, response)
                            relevance = st.session_state.chatbot_rag.evaluate_relevance(query, response)
                            context_precision = st.session_state.chatbot_rag.evaluate_context_precision(query, docs)
                            
                            results.append({
                                'query': query,
                                'response': response[:100] + "...",
                                'retrieval_time': retrieval_time,
                                'generation_time': generation_time,
                                'faithfulness': faithfulness,
                                'relevance': relevance,
                                'context_precision': context_precision,
                                'ground_truth': test_case['ground_truth'][:100] + "..."
                            })
                
                if results:
                    st.subheader("📊 Resultados de Evaluación")
                    eval_df = pd.DataFrame(results)
                    st.dataframe(eval_df)
                    
                    st.subheader("📈 Métricas Promedio")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Fidelidad", f"{eval_df['faithfulness'].mean():.1f}/10")
                    with col2:
                        st.metric("Relevancia", f"{eval_df['relevance'].mean():.1f}/10")
                    with col3:
                        st.metric("Precisión", f"{eval_df['context_precision'].mean():.2f}")
                    with col4:
                        st.metric("Tiempo total", f"{(eval_df['retrieval_time'] + eval_df['generation_time']).mean():.2f}s")
    
    with tab5:
        st.header("📈 Analytics y Exportación")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📤 Exportar Datos")
            
            if st.button("📊 Exportar para LangSmith"):
                if st.session_state.chatbot_rag.interaction_logs:
                    langsmith_data = export_langsmith_format(st.session_state.chatbot_rag.interaction_logs)
                    st.json(langsmith_data[:2])  # Preview
                    
                    json_str = json.dumps(langsmith_data, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="💾 Descargar JSON LangSmith",
                        data=json_str,
                        file_name=f"langsmith_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                else:
                    st.info("No hay datos para exportar")
            
            if st.button("📊 Exportar CSV"):
                if st.session_state.chatbot_rag.interaction_logs:
                    df = pd.DataFrame([
                        {
                            'timestamp': log['timestamp'],
                            'query': log['query'],
                            'response': log['response'][:100] + "...",
                            **log['metrics']
                        }
                        for log in st.session_state.chatbot_rag.interaction_logs
                    ])
                    
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="💾 Descargar CSV",
                        data=csv,
                        file_name=f"rag_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No hay datos para exportar")
        
        with col2:
            st.subheader("📊 Análisis de Documentos")
            
            if st.session_state.chatbot_rag.documents:
                # Document statistics
                doc_lengths = [len(doc) for doc in st.session_state.chatbot_rag.documents]
                
                fig = px.bar(
                    x=list(range(1, len(doc_lengths) + 1)),
                    y=doc_lengths,
                    title="Distribución de Longitud de Documentos",
                    labels={'x': 'ID Documento', 'y': 'Caracteres'}
                )
                st.plotly_chart(fig, use_container_width=True)

                # Frecuencia de palabras
                all_text = " ".join(st.session_state.chatbot_rag.documents)
                words = all_text.lower().split()
                word_freq = {}
                for word in words:
                    if len(word) > 3:  # Filtra palabras cortas
                        word_freq[word] = word_freq.get(word, 0) + 1
                
                top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
                
                if top_words:
                    fig = px.bar(
                        x=[word[0] for word in top_words],
                        y=[word[1] for word in top_words],
                        title="Top 10 Palabras Más Frecuentes"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay documentos disponibles para análisis")

if __name__ == "__main__":
    main()