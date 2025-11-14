#!/usr/bin/env python3
"""
Socket.IO Voice Server for Frontend Integration
Works with Next.js frontend using socket.io-client
"""

import socketio
import logging
import asyncio
from aiohttp import web
from voice_chatbot import VoiceChatbot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Socket.IO server with CORS enabled
sio = socketio.AsyncServer(
    cors_allowed_origins='*',  # Allow all origins for development
    async_mode='aiohttp',
    logger=True,
    engineio_logger=True
)

app = web.Application()
sio.attach(app)

# Store client sessions
clients = {}

@sio.event
async def connect(sid, environ):
    """Handle client connection"""
    logger.info(f"🔗 Client {sid} connected")
    
    try:
        # Create individual session for this client
        chatbot = VoiceChatbot()
        session_id = chatbot.rag_orchestrator.create_session()
        chatbot.session_id = session_id
        
        clients[sid] = {
            'chatbot': chatbot,
            'session_id': session_id
        }
        
        # Send simple greeting (no session ID)
        await sio.emit('message', {
            'type': 'connected',
            'data': 'Hello! I\'m ready to help you. Start speaking anytime!'
        }, room=sid)
        
        logger.info(f"✅ Session {session_id} created for client {sid}")
        
    except Exception as e:
        logger.error(f"Error creating session for {sid}: {e}")
        await sio.emit('message', {
            'type': 'error',
            'data': f'Failed to create session: {str(e)}'
        }, room=sid)

@sio.event
async def disconnect(sid):
    """Handle client disconnection"""
    logger.info(f"🔌 Client {sid} disconnected")
    
    if sid in clients:
        session_id = clients[sid]['session_id']
        del clients[sid]
        logger.info(f"🗑️ Session {session_id} cleaned up for client {sid}")

@sio.event
async def query(sid, data):
    """Handle query from client"""
    try:
        if sid not in clients:
            await sio.emit('message', {
                'type': 'error',
                'data': 'Session not found'
            }, room=sid)
            return
        
        # Extract query text
        if isinstance(data, dict):
            query_text = data.get('query', '')
        else:
            query_text = str(data)
        
        if not query_text or not query_text.strip():
            await sio.emit('message', {
                'type': 'error',
                'data': 'Empty query'
            }, room=sid)
            return
        
        client_data = clients[sid]
        chatbot = client_data['chatbot']
        session_id = client_data['session_id']
        
        logger.info(f"🗣️ Client {sid} | Session {session_id}: '{query_text}'")
        
        # Send status update
        await sio.emit('message', {
            'type': 'status',
            'data': f'Processing: {query_text}'
        }, room=sid)
        
        # Collect responses
        responses = []
        
        # Override speak_text to capture responses
        original_speak = chatbot.speak_text
        
        def capture_speak(text):
            # Just collect responses, we'll send them later
            responses.append(text)
            logger.info(f"📝 Captured response: {text[:50]}...")
        
        chatbot.speak_text = capture_speak
        
        try:
            # Process query in background
            logger.info(f"🔄 Processing with session {session_id}...")
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                None,
                chatbot.process_voice_query,
                query_text
            )
            
            # Give extra time for background threads
            await asyncio.sleep(1)
            
            logger.info(f"✅ Client {sid} processing completed with {len(responses)} responses")
            
            # Send all collected responses
            for response_text in responses:
                await sio.emit('message', {
                    'type': 'response',
                    'data': response_text
                }, room=sid)
                logger.info(f"🔊 Sent to client {sid}: {response_text[:50]}...")
            
            await sio.emit('message', {
                'type': 'status',
                'data': f'✅ Completed - Ready for next question'
            }, room=sid)
            
        finally:
            # Restore original speak
            chatbot.speak_text = original_speak
            
    except Exception as e:
        logger.error(f"Error processing query for {sid}: {e}")
        await sio.emit('message', {
            'type': 'error',
            'data': str(e)
        }, room=sid)

# Health check endpoint
async def health(request):
    """Health check endpoint"""
    return web.Response(text='Socket.IO Server Running')

# Add routes
app.router.add_get('/', health)
app.router.add_get('/health', health)

def main():
    """Start the Socket.IO server"""
    logger.info("🚀 Starting Socket.IO Server on http://localhost:8889")
    logger.info("👥 Each client gets their own session")
    logger.info("🗑️ Sessions expire when clients disconnect")
    logger.info("🌐 CORS enabled for frontend integration")
    logger.info("-" * 50)
    
    web.run_app(app, host='localhost', port=8889)
    # web.run_app(app, host='0.0.0.0', port=8889)

if __name__ == "__main__":
    main()

