import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

export interface ExportRequest {
  composition_id: string;
  format: 'wav' | 'flac' | 'mp3' | 'midi';
  quality: 'standard' | 'high' | 'ultra';
}

export const exportComposition = async (request: ExportRequest) => {
  const response = await axios.post(`${API_BASE_URL}/premium/export`, request);
  return response.data;
};

export const getSubscriptionInfo = async () => {
  const response = await axios.get(`${API_BASE_URL}/premium/subscription`);
  return response.data;
};

export const upgradeSubscription = async (plan: string) => {
  const response = await axios.post(`${API_BASE_URL}/premium/upgrade`, null, {
    params: { plan },
  });
  return response.data;
};

export const getAIModels = async () => {
  const response = await axios.get(`${API_BASE_URL}/premium/ai-models`);
  return response.data;
};

export const generateWithAI = async (
  origin: string,
  destination: string,
  model_id: string,
  priority: boolean = false
) => {
  const response = await axios.post(`${API_BASE_URL}/premium/generate-with-ai`, null, {
    params: { origin, destination, model_id, priority },
  });
  return response.data;
};
