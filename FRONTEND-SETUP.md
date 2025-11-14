# 🚀 Complete Setup Guide - Frontend + Backend

This guide will help you run the complete Voice AI Chatbot with the Next.js frontend.

## 📋 Prerequisites

- Python 3.8+ installed
- Node.js 18+ installed
- Redis server installed and running
- OpenAI API key

---

## 🎯 Quick Start (3 Steps)

### **Step 1: Start Redis**

```bash
# Option A: Windows with WSL
wsl
sudo service redis-server start

# Option B: Direct Windows Redis
redis-server
```

### **Step 2: Start Backend (Socket.IO Server)**

Open a **new terminal** in the project root:

```bash
# Make sure Redis is running first!
py socketio_server.py
```

You should see:
```
🚀 Starting Socket.IO Server on http://localhost:8889
👥 Each client gets their own session
🗑️ Sessions expire when clients disconnect
🌐 CORS enabled for frontend integration
```

### **Step 3: Start Frontend**

Open **another new terminal**:

```bash
cd frontend
npm run dev
```

Frontend will be available at: `http://localhost:3000`

---

## 📁 Project Structure

```
voice-chatbot-main/
├── frontend/                    # Next.js frontend
│   ├── app/
│   │   └── page.tsx            # Home page with button
│   ├── components/
│   │   └── VoiceChatbot.tsx    # Main chatbot UI
│   └── package.json
│
├── agents/                      # AI agents
├── vectorstore/                 # ChromaDB
├── utils/                       # Redis manager
├── data/                        # Privacy policy data
│
├── socketio_server.py          # ✨ NEW: Socket.IO server for frontend
├── socket_server.py            # OLD: TCP socket server
├── voice_chatbot.py            # Voice chatbot logic
├── main.py                     # CLI version
├── initialize_data.py          # Database setup
└── requirements.txt
```

---

## 🔧 First Time Setup

### 1. Setup Backend

```bash
# Install Python dependencies
pip install -r requirements.txt
pip install python-socketio aiohttp

# Create .env file
echo "OPENAI_API_KEY=your-actual-api-key-here" > .env
echo "REDIS_HOST=localhost" >> .env
echo "REDIS_PORT=6379" >> .env

# Initialize database
python initialize_data.py --reset
```

### 2. Setup Frontend

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Done! No .env needed for development
```

---

## 🎮 Usage

### Option A: Web Interface (Recommended)

1. **Start Backend**: `py socketio_server.py`
2. **Start Frontend**: `cd frontend && npm run dev`
3. **Open Browser**: `http://localhost:3000`
4. **Click Button**: "Start Voice Chat"
5. **Type or Record**: Send your questions!

### Option B: CLI Version (No Frontend)

```bash
# Text-based chatbot
py main.py

# Voice-based chatbot (with TTS)
py voice_chatbot.py
```

---

## 🌐 How It Works

### Architecture Flow

```
User Browser (http://localhost:3000)
    ↓
Next.js Frontend
    ↓ [Socket.IO]
Socket.IO Server (port 8889)
    ↓
Voice Chatbot Logic
    ↓
┌──────────────┬────────────────┐
│ Redis Cache  │ ChromaDB RAG   │
└──────────────┴────────────────┘
```

### Communication Protocol

**Frontend → Backend:**
```javascript
// Connect
socket.connect();

// Send query
socket.emit('query', { query: 'What is your privacy policy?' });
```

**Backend → Frontend:**
```javascript
// Connection confirmed
{ type: 'connected', data: 'Connected! Your session ID: xxx' }

// Processing status
{ type: 'status', data: 'Processing: your question' }

// AI Response (streaming)
{ type: 'response', data: 'Here is the answer...' }

// Completion
{ type: 'status', data: '✅ Completed - Ready for next question' }
```

---

## 🔍 Testing the Setup

### Test Backend

```bash
# Terminal 1: Start backend
py socketio_server.py

# You should see server starting message
```

### Test Frontend

```bash
# Terminal 2: Start frontend
cd frontend
npm run dev

# Open http://localhost:3000 in browser
# You should see the home page with "Start Voice Chat" button
```

### Test Full Flow

1. Click "Start Voice Chat" button
2. Wait for "Connected!" status
3. Type a question: "What is your privacy policy?"
4. Click "Send"
5. See the response appear in real-time!

---

## 🐛 Troubleshooting

### Backend Issues

**Problem**: `ModuleNotFoundError: No module named 'socketio'`

**Solution**:
```bash
pip install python-socketio aiohttp
```

---

**Problem**: `Redis connection failed`

**Solution**:
```bash
# Check if Redis is running
redis-cli ping  # Should return "PONG"

# If not running, start it
redis-server
```

---

**Problem**: Port 8889 already in use

**Solution**:
```bash
# Find process using port
netstat -ano | findstr :8889

# Kill the process (Windows)
taskkill /PID <process_id> /F

# Or use a different port in both files:
# - socketio_server.py: web.run_app(app, port=8890)
# - frontend/components/VoiceChatbot.tsx: io('http://localhost:8890')
```

---

### Frontend Issues

**Problem**: `Connection error. Make sure backend server is running`

**Solution**:
1. Verify backend is running: `py socketio_server.py`
2. Check console for errors (F12 in browser)
3. Verify URL in `VoiceChatbot.tsx` matches backend port

---

**Problem**: Microphone not working

**Solution**:
- Use the text input instead (type your questions)
- Allow microphone access in browser settings
- For now, voice recognition uses a prompt (production will use proper API)

---

**Problem**: Build errors with npm

**Solution**:
```bash
# Clear everything and reinstall
cd frontend
rm -rf node_modules .next
npm install
npm run dev
```

---

## 📊 Monitoring

### Backend Logs

Watch the `socketio_server.py` terminal for:
- `🔗 Client {sid} connected` - New connection
- `🗣️ Client {sid} | Session {session_id}: 'query'` - Query received
- `🔊 Client {sid}: response...` - Response sent
- `✅ Client {sid} processing completed` - Done

### Frontend Logs

Open browser console (F12) to see:
- Socket connection status
- Messages sent/received
- Any errors

---

## 🚀 Production Deployment

### Backend (Python)

```bash
# Use gunicorn with workers
pip install gunicorn
gunicorn socketio_server:app --worker-class aiohttp.GunicornWebWorker --workers 4 --bind 0.0.0.0:8889
```

### Frontend (Next.js)

```bash
# Build for production
cd frontend
npm run build
npm start

# Or deploy to Vercel
vercel --prod
```

---

## 🎨 Customization

### Change Colors

Edit `frontend/app/page.tsx` and `frontend/components/VoiceChatbot.tsx`:
- Replace `blue-600` with your color
- Replace `indigo-600` with your accent color

### Change Recording Duration

Edit `frontend/components/VoiceChatbot.tsx` line ~122:
```typescript
setTimeout(() => {
  stopRecording();
}, 8000); // Change to 10000 for 10 seconds
```

### Add Speech Recognition

Replace the prompt in `transcribeAudio` function with:
- Google Cloud Speech-to-Text API
- Azure Speech Services
- Browser Web Speech API

---

## 📝 Environment Variables

### Backend (.env)

```env
OPENAI_API_KEY=your-key-here
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
```

### Frontend (.env.local) - Optional

```env
NEXT_PUBLIC_SOCKET_URL=http://localhost:8889
```

---

## ✅ Complete Checklist

- [ ] Redis installed and running
- [ ] Python dependencies installed
- [ ] OpenAI API key set in .env
- [ ] Database initialized with `initialize_data.py`
- [ ] Backend running on port 8889
- [ ] Frontend running on port 3000
- [ ] Can see home page in browser
- [ ] Can connect to chatbot
- [ ] Can send messages and receive responses

---

## 💡 Tips

1. **Keep Redis Running**: The backend needs Redis to work
2. **Watch Logs**: Terminal logs show what's happening
3. **Browser Console**: F12 to see frontend logs
4. **Port Numbers**:
   - Frontend: 3000
   - Backend Socket.IO: 8889
   - Redis: 6379

---

## 🎯 Next Steps

1. **Test the basic flow**: Home page → Button → Chat → Send message
2. **Customize the UI**: Change colors, text, styling
3. **Add features**: More questions, better UI, error handling
4. **Deploy**: Put it online for others to use!

---

**Need Help?**
- Check logs in both terminals
- Use browser console (F12)
- Verify all services are running
- Review the README files in each directory

---

**🎉 You're all set! Click that button and start chatting!**

