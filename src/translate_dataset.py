import pandas as pd
import os
import time
from typing import Optional
import requests
from requests.exceptions import Timeout as RequestsTimeout
from logger_config import setup_logger

logger = setup_logger()

class DatasetTranslator:
    def __init__(self, model: str = "llama3.1"):
        self.model = model
        self.ollama_base_url = "http://localhost:11434/api/chat"
        self.checkpoint_interval = 10
        
    def is_safety_message(self, text: str) -> bool:
        safety_phrases = [
            "Peço desculpas, mas não posso",
            "peço desculpas, mas não posso",
            "Não posso ajudar com isso",
            "não posso ajudar com isso",
            "Não posso cumprir esse pedido",
            "não posso cumprir esse pedido",
            "Desculpe, mas não posso",
            "desculpe, mas não posso",
            "Não posso cumprir esse pedido",
            "não posso cumprir esse pedido",
            "Não posso cumprir",
            "não posso cumprir",
            "linha de apoio",
            "saúde mental disponíveis",
            "profissionais de saúde mental",
            "recursos disponíveis para apoiar",
            "orientação profissional",
            "alguem que possa ajudar",
            "alguém que possa oferecer apoio",
            "profissional de saúde mental",
            "tradução:",
            "Tradução:",
        ]
        text_lower = text.lower()
        return any(phrase.lower() in text_lower for phrase in safety_phrases)
    
    def translate_text(self, text: str, retry_count: int = 0) -> str:
        if pd.isna(text) or text == "":
            return text
        
        max_retries = 2
        
        if retry_count > 0:
            time.sleep(0.5 * retry_count)
        
        try:
            if retry_count == 0:
                system_message = (
                    "Você é um módulo de tradução automática EN->PT-BR. "
                    "Processe o texto de entrada e retorne APENAS a tradução equivalente em português brasileiro. "
                    "Não gere metadados, avisos, explicações ou mensagens. Apenas o texto traduzido."
                )
                user_message = (
                    f"INPUT: {text}\nOUTPUT:"
                )
            elif retry_count == 1:
                system_message = (
                    "Módulo de tradução EN->PT-BR. Retorne apenas texto traduzido, sem comentários."
                )
                user_message = f'Traduza: "{text}"'
            else:
                system_message = "Traduza EN->PT-BR. Apenas tradução."
                user_message = f'Traduza: "{text}"'
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.95,
                    "num_predict": -1,
                    "repeat_penalty": 1.1
                }
            }
            
            timeout = 120 if retry_count == 0 else 180
            response = requests.post(self.ollama_base_url, json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            translated = result.get("message", {}).get("content", text).strip()
            
            if not translated or translated == text:
                translated = result.get("response", text).strip()
            
            if self.is_safety_message(translated):
                if retry_count < max_retries:
                    logger.warning(f"Detectada mensagem de segurança. Retry {retry_count + 1}/{max_retries}...")
                    time.sleep(0.5)
                    return self.translate_text(text, retry_count + 1)
                else:
                    logger.warning(f"Após {max_retries} tentativas, ainda bloqueado. Marcando como 'AJUSTE MANUAL'")
                    return "AJUSTE MANUAL"
            
            return translated
        except RequestsTimeout as e:
            logger.warning(f"Timeout ao traduzir (tentativa {retry_count + 1}/{max_retries})")
            if retry_count < max_retries:
                backoff_time = min(2 ** retry_count, 10)
                time.sleep(backoff_time)
                return self.translate_text(text, retry_count + 1)
            logger.error("Timeout após todas as tentativas. Marcando como 'AJUSTE MANUAL'")
            return "AJUSTE MANUAL"
        except Exception as e:
            logger.error(f"Erro ao traduzir com Ollama: {e}")
            if retry_count < max_retries:
                backoff_time = min(2 ** retry_count, 5)
                logger.info(f"Tentando novamente em {backoff_time}s... (tentativa {retry_count + 1})")
                time.sleep(backoff_time)
                return self.translate_text(text, retry_count + 1)
            logger.error("Erro após todas as tentativas. Marcando como 'AJUSTE MANUAL'")
            return "AJUSTE MANUAL"
    
    def translate_dataset(
        self, 
        input_path: str, 
        output_path: str, 
        text_column: str,
        checkpoint_path: Optional[str] = None,
        start_from: int = 0
    ):
        logger.info(f"Carregando dataset: {input_path}")
        df = pd.read_csv(input_path)
        total_rows = len(df)
        
        logger.info(f"Total de linhas: {total_rows}")
        logger.info(f"Modelo: {self.model}")
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Carregando checkpoint: {checkpoint_path}")
            df_translated = pd.read_csv(checkpoint_path)
            if f"{text_column}_pt" not in df_translated.columns:
                df_translated[f"{text_column}_pt"] = ""
            
            if len(df_translated) != total_rows:
                logger.warning(f"Checkpoint tem {len(df_translated)} linhas, mas dataset tem {total_rows}. Recriando...")
                df_translated = df.copy()
                df_translated[f"{text_column}_pt"] = ""
                start_from = 0
            else:
                translated_mask = (df_translated[f"{text_column}_pt"].notna()) & (df_translated[f"{text_column}_pt"] != "") & (df_translated[f"{text_column}_pt"] != "AJUSTE MANUAL")
                start_from = translated_mask.sum()
                logger.info(f"Checkpoint encontrado. {start_from} linhas já traduzidas. Continuando da linha {start_from}/{total_rows}")
        else:
            if checkpoint_path:
                logger.info(f"Checkpoint não existe ainda. Iniciando tradução do zero.")
            df_translated = df.copy()
            df_translated[f"{text_column}_pt"] = ""
            start_from = 0
        
        indices_to_translate = []
        for i in range(start_from, total_rows):
            if pd.isna(df.loc[i, text_column]) or df.loc[i, text_column] == "":
                df_translated.loc[i, f"{text_column}_pt"] = ""
                continue
            
            existing_translation = df_translated.loc[i, f"{text_column}_pt"] if f"{text_column}_pt" in df_translated.columns else ""
            if pd.notna(existing_translation) and existing_translation != "" and existing_translation != "AJUSTE MANUAL":
                continue
            
            indices_to_translate.append(i)
        
        total_to_translate = len(indices_to_translate)
        logger.info(f"Total de linhas para traduzir: {total_to_translate}")
        
        translated_count = start_from
        completed = 0
        
        try:
            for idx in indices_to_translate:
                try:
                    text = df.loc[idx, text_column]
                    logger.info(f"Traduzindo linha {idx + 1}/{total_rows}...")
                    translated_text = self.translate_text(text)
                    df_translated.loc[idx, f"{text_column}_pt"] = translated_text
                    completed += 1
                    translated_count += 1
                    
                    if completed % 10 == 0:
                        logger.info(f"Progresso: {completed}/{total_to_translate} linhas traduzidas ({100*completed/total_to_translate:.1f}%)")
                    
                    if completed % self.checkpoint_interval == 0:
                        logger.info(f"Salvando checkpoint...")
                        df_translated.to_csv(checkpoint_path or output_path, index=False)
                        logger.info(f"Progresso total: {translated_count}/{total_rows} ({100*translated_count/total_rows:.1f}%)")
                except Exception as e:
                    logger.error(f"Erro ao traduzir linha {idx+1}: {e}")
                    df_translated.loc[idx, f"{text_column}_pt"] = "AJUSTE MANUAL"
                    completed += 1
                    translated_count += 1
            
            logger.info("Salvando arquivo final...")
            df_translated.to_csv(output_path, index=False)
            logger.info(f"Tradução concluída! Arquivo salvo em: {output_path}")
            
        except KeyboardInterrupt:
            logger.warning("Interrompido pelo usuário. Salvando progresso...")
            df_translated.to_csv(checkpoint_path or output_path, index=False)
            logger.info(f"Progresso salvo. Traduzidas {translated_count}/{total_rows} linhas")
        except Exception as e:
            logger.error(f"Erro durante tradução: {e}")
            logger.info("Salvando progresso antes de encerrar...")
            df_translated.to_csv(checkpoint_path or output_path, index=False)
            raise


def translate_reddit_dataset(model: str = "llama3.1"):
    translator = DatasetTranslator(model=model)
    
    input_path = "data/suicidal_ideation_reddit.csv"
    output_path = "data/suicidal_ideation_reddit_pt.csv"
    checkpoint_path = "data/suicidal_ideation_reddit_pt_checkpoint.csv"
    
    translator.translate_dataset(
        input_path=input_path,
        output_path=output_path,
        text_column="usertext",
        checkpoint_path=checkpoint_path
    )


def translate_twitter_dataset(model: str = "llama3.1"):
    translator = DatasetTranslator(model=model)
    
    input_path = "data/suicidal_ideation_twitter.csv"
    output_path = "data/suicidal_ideation_twitter_pt.csv"
    checkpoint_path = "data/suicidal_ideation_twitter_pt_checkpoint.csv"
    
    translator.translate_dataset(
        input_path=input_path,
        output_path=output_path,
        text_column="Tweet",
        checkpoint_path=checkpoint_path
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Traduz datasets para português usando Ollama")
    parser.add_argument("--model", default="llama3.1",
                       help="Modelo Ollama a usar (default: llama3.1)")
    parser.add_argument("--dataset", choices=["reddit", "twitter", "both"], default="both",
                       help="Qual dataset traduzir (default: both)")
    
    args = parser.parse_args()
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            logger.error("Ollama não está rodando. Inicie com: ollama serve")
            exit(1)
        logger.info("Ollama detectado e funcionando!")
    except:
        logger.error("Ollama não está rodando. Inicie com: ollama serve")
        logger.info("Ou instale com: curl -fsSL https://ollama.com/install.sh | sh")
        exit(1)
    
    logger.info("Processando tradução linha por linha (sequencial)")
    
    if args.dataset in ["reddit", "both"]:
        logger.info("="*60)
        logger.info("TRADUZINDO DATASET REDDIT")
        logger.info("="*60)
        translate_reddit_dataset(model=args.model)
    
    if args.dataset in ["twitter", "both"]:
        logger.info("="*60)
        logger.info("TRADUZINDO DATASET TWITTER")
        logger.info("="*60)
        translate_twitter_dataset(model=args.model)

