from rest_framework import serializers

class ChatInputSerializer(serializers.Serializer):
    user_input = serializers.CharField()

class ChatOutputSerializer(serializers.Serializer):
    ai_response = serializers.CharField()