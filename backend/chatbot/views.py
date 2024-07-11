from pathlib import Path
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response

from .chatbot import get_query_response

# Create your views here.

class QueryAPIView(APIView):
    def post(self, request):
        try:
            query = request.data.get('query')
            if query and query.strip():
                file_path = Path(__file__).parent / 'document.pdf'
                query_response = get_query_response(file_path=file_path, query=query, session_id='abc123')
                return Response(data={'query_response': query_response}, status=status.HTTP_200_OK)
            else:
                return Response(data={
                    'warning': 'Missing Query Input',
                    'message': 'The query field is required and cannot be empty.'
                }, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response(data={
                'error': 'Internal Server Error',
                'message': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)