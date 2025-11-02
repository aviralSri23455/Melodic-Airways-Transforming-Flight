import { apiClient, ApiResponse } from './client';

export interface Lesson {
  id: string;
  title: string;
  description: string;
  type: string;
  difficulty: string;
}

export interface LessonStartResponse {
  lesson_id: string;
  content: any;
  interactive_data: any;
}

export const getLessons = async (): Promise<ApiResponse<Lesson[]>> => {
  return await apiClient.get<Lesson[]>('/education/lessons');
};

export const startLesson = async (
  lessonId: string,
  lessonType: string,
  difficulty: string
): Promise<ApiResponse<LessonStartResponse>> => {
  return await apiClient.post<LessonStartResponse>(
    `/education/lessons/${lessonId}/start`,
    {
      lesson_type: lessonType,
      difficulty,
    }
  );
};

export const visualizeGraphAlgorithm = async (
  origin: string,
  destination: string
): Promise<ApiResponse<any>> => {
  return await apiClient.get<any>(
    `/education/graph-visualization/${origin}/${destination}`
  );
};
