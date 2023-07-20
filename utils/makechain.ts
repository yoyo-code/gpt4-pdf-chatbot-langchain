import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

const CONDENSE_PROMPT = `Dada la siguiente conversación y una pregunta de seguimiento, reformule la pregunta de seguimiento para que sea una pregunta independiente

Historia de conversacion:
{chat_history}
Pregunta de seguimiento: {question}
Pregunta independiente:`;

const QA_PROMPT = `Eres un experto asistente sobre las patologias GES en Chile. Con ayuda de las piezas de contexto obtenidas del manual de parametrizacion de patologias GES responde la pregunta al final.
Si no sabes la respuesta, usa tus propios conocimientos sobre las patologias GES en Chile para responder de forma clara y amigable.
Si la pregunta no está relacionada con el contexto, responde cortésmente que estás entrenado para responder solo preguntas relacionadas al contexto.
Si encuentras que es necesaria mas informacion para dar una respuesta puedes preguntar por mas informacion.

Piezas de contexto: {context}

Pregunta: {question}
Respuesta útil en Markdown:`;

export const makeChain = (vectorstore: PineconeStore) => {
  const model = new OpenAI({
    temperature: 0.7,
    modelName: 'gpt-4',
    topP: 1,
    frequencyPenalty: 0,
    presencePenalty: 0,
    maxTokens: 2048,
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(),
    {
      qaTemplate: QA_PROMPT,
      questionGeneratorTemplate: CONDENSE_PROMPT,
      returnSourceDocuments: false,
    },
  );
  return chain;
};
