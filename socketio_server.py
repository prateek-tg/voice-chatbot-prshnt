#!/usr/bin/env python3
"""
Socket.IO Text Server for Frontend Integration
Works with Next.js frontend using socket.io-client
"""

import socketio
import logging
import asyncio
from aiohttp import web
from text_chatbot import TextChatbot

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
async def connect(sid, environ, auth=None):
    """Handle client connection with optional auth data"""
    logger.info(f"🔗 Client {sid} connected")
    if auth:
        logger.info(f"📋 Connection data: {auth}")
    
    try:
        # Create individual session for this client
        chatbot = TextChatbot()
        session_id = chatbot.create_session()
        
        clients[sid] = {
            'chatbot': chatbot,
            'session_id': session_id
        }
        
        # Send simple greeting (no session ID)
        await sio.emit('message', {
            'type': 'connected',
            'data': 'Hello! I\'m ready to help you. Start typing anytime!'
        }, room=sid)
        
        logger.info(f"✅ Session {session_id} created for client {sid}")
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error creating session for {sid}: {e}")
        
        # Provide helpful error message for common issues
        if "Could not connect to tenant" in error_message or "default_tenant" in error_message:
            helpful_message = "Server database not initialized. Please contact the administrator to run: python3 initialize_data.py --reset"
        else:
            helpful_message = f'Failed to create session: {error_message}'
        
        await sio.emit('message', {
            'type': 'error',
            'data': helpful_message
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
        
        logger.info(f"� Client {sid} | Session {session_id}: '{query_text}'")
        
        # Send status update
        await sio.emit('message', {
            'type': 'status',
            'data': f'Processing: {query_text}'
        }, room=sid)
        
        try:
            # Process query using text_chatbot method
            logger.info(f"🔄 Processing text query with session {session_id}...")
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                chatbot.process_text_query,
                query_text
            )
            
            logger.info(f"✅ Client {sid} processing completed")
            
            # Handle different response types
            if response:
                response_text = response.get('response', '')
                response_type = response.get('response_type', 'unknown')
                should_continue = response.get('should_continue', True)
                
                if response_text:
                    # Send the main response
                    await sio.emit('message', {
                        'type': 'response',
                        'data': response_text
                    }, room=sid)
                    logger.info(f"� Sent to client {sid}: {response_text[:50]}...")
                
                # If there's a detailed response (for intermediate responses)
                if response.get('has_detailed_response') and response_type == 'intermediate':
                    # Check if detailed response is already available
                    detailed_response = response.get('detailed_response')
                    if detailed_response:
                        await sio.emit('message', {
                            'type': 'response',
                            'data': detailed_response
                        }, room=sid)
                        logger.info(f"� Sent detailed response to client {sid}: {detailed_response[:50]}...")
                    else:
                        # Get detailed response asynchronously
                        try:
                            detailed_resp = await loop.run_in_executor(
                                None,
                                chatbot.get_detailed_response,
                                query_text
                            )
                            if detailed_resp and detailed_resp.get('response'):
                                await sio.emit('message', {
                                    'type': 'response',
                                    'data': detailed_resp['response']
                                }, room=sid)
                                logger.info(f"📤 Sent delayed detailed response to client {sid}: {detailed_resp['response'][:50]}...")
                        except Exception as e:
                            logger.error(f"Error getting detailed response for {sid}: {e}")
                
                # Send completion status
                if should_continue:
                    await sio.emit('message', {
                        'type': 'status',
                        'data': '✅ Completed - Ready for next question'
                    }, room=sid)
                else:
                    # User indicated end of conversation
                    await sio.emit('message', {
                        'type': 'status',
                        'data': '👋 Conversation ended'
                    }, room=sid)
            else:
                await sio.emit('message', {
                    'type': 'error',
                    'data': 'No response received'
                }, room=sid)
            
        except Exception as e:
            logger.error(f"Error processing text query for {sid}: {e}")
            await sio.emit('message', {
                'type': 'error',
                'data': f'Processing error: {str(e)}'
            }, room=sid)
            
    except Exception as e:
        logger.error(f"Error handling query for {sid}: {e}")
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
    logger.info("🚀 Starting Socket.IO Text Server on http://localhost:8889")
    logger.info("👥 Each client gets their own session")
    logger.info("🗑️ Sessions expire when clients disconnect")
    logger.info("💬 Optimized for text-based interactions")
    logger.info("🌐 CORS enabled for frontend integration")
    logger.info("-" * 50)
    
    web.run_app(app, host='localhost', port=8889)
    # web.run_app(app, host='0.0.0.0', port=8889)

if __name__ == "__main__":
    main()

