import React, { useState, useEffect } from 'react';
import {
  AppBar, Tabs, Tab, Toolbar, Typography, Grid, Paper, Box, Button, Select, MenuItem, TextField, CircularProgress, Divider
} from '@mui/material';
import axios from 'axios';
import SendIcon from '@mui/icons-material/RocketLaunch';
import ArrowDropUpIcon from '@mui/icons-material/ArrowDropUp';
import InputAdornment from '@mui/material/InputAdornment';
import IconButton from '@mui/material/IconButton';
import Popover from '@mui/material/Popover';
import ReactMarkdown from 'react-markdown';
import Autocomplete from '@mui/material/Autocomplete';

const API_URL = 'http://localhost:8000';

function App() {
  const [tab, setTab] = useState(0); // 0: Query, 1: MCQ, 2: Upload
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [domains, setDomains] = useState([]);
  const [selectedDomain, setSelectedDomain] = useState('');
  const [loading, setLoading] = useState(false);

  // Query Response states
  const [query, setQuery] = useState('');
  const [queryResponse, setQueryResponse] = useState(null);

  // MCQ states
  const [numQuestions, setNumQuestions] = useState(5);
  const [difficulty, setDifficulty] = useState('Medium');
  const [mcqs, setMcqs] = useState(null);
  const [topic, setTopic] = useState('');
  const [mcqView, setMcqView] = useState(0); // 0: Student, 1: Teacher

  // File Upload states
  const [files, setFiles] = useState([]);
  const [subject, setSubject] = useState('');
  const [uploadDomain, setUploadDomain] = useState('');

  // For domain selection popover
  const [anchorEl, setAnchorEl] = useState(null);

  // Chat history state
  const [chatHistory, setChatHistory] = useState([]); // {role: 'user'|'model', text: string, sources?: []}

  const chatEndRef = React.useRef(null);
  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [chatHistory]);

  useEffect(() => {
    // Fetch available models and domains
    fetchModels();
    fetchDomains();
  }, []);

  const fetchModels = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/models`);
      setModels(response.data.models);
    } catch (error) {
      console.error('Error fetching models:', error);
    }
  };

  const fetchDomains = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/domains`);
      setDomains(response.data.domains);
    } catch (error) {
      console.error('Error fetching domains:', error);
    }
  };

  const handleQuery = async () => {
    if (!query.trim()) return;
    setLoading(true);
    // Add user message to chat
    setChatHistory(prev => [...prev, { role: 'user', text: query }]);
    try {
      const response = await axios.post(`${API_URL}/api/query`, {
        query,
        domain: selectedDomain
      });
      setChatHistory(prev => [
        ...prev,
        {
          role: 'model',
          text: response.data.answer,
          sources: response.data.sources
        }
      ]);
      setQuery('');
    } catch (error) {
      setChatHistory(prev => [
        ...prev,
        { role: 'model', text: "Sorry, there was an error processing your request." }
      ]);
    }
    setLoading(false);
  };

  const handleMCQ = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API_URL}/api/mcq`, {
        domain: selectedDomain,
        num_questions: numQuestions,
        difficulty,
        model_name: "DeepSeek", // Hardcoded
        topic
      });
      setMcqs(response.data.mcqs);
    } catch (error) {
      console.error('Error generating MCQs:', error);
    }
    setLoading(false);
  };

  const handleFileUpload = async () => {
    setLoading(true);
    try {
      const formData = new FormData();
      files.forEach(file => formData.append('files', file));
      formData.append('domain', uploadDomain);
      formData.append('subject', subject);

      await axios.post(`${API_URL}/api/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      alert('Files uploaded successfully!');
      setFiles([]);
      setSubject('');
      setUploadDomain('');
    } catch (error) {
      console.error('Error uploading files:', error);
      alert('Error uploading files');
    }
    setLoading(false);
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      {/* Top AppBar with Tabs */}
      <AppBar position="static" color="primary" elevation={2}>
        <Toolbar>
          <Typography variant="h6" sx={{ flexGrow: 1 }}>
            RAG Based Learning Assistant
          </Typography>
          <Tabs value={tab} onChange={(e, v) => setTab(v)} textColor="inherit" indicatorColor="secondary">
            <Tab label="Query" />
            <Tab label="MCQ Generator" />
            <Tab label="File Upload" />
          </Tabs>
        </Toolbar>
      </AppBar>

      {/* Main Content */}
      <Box sx={{ p: 3 }}>
        {tab === 0 && (
          <Box sx={{ height: '70vh', display: 'flex', flexDirection: 'column', justifyContent: 'flex-end', alignItems: 'center' }}>
            {/* Chat-like response area */}
            <Paper sx={{ width: '100%', maxWidth: 700, mb: 2, p: 3, overflowY: 'auto', flexGrow: 1, minHeight: 300 }}>
              <Typography variant="h5" gutterBottom>Query Response</Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                {chatHistory.length === 0 && (
                  <Typography variant="body2" color="text.secondary">
                    Ask a question to get started!
                  </Typography>
                )}
                {chatHistory.map((msg, idx) => (
                  <Box
                    key={idx}
                    sx={{
                      alignSelf: msg.role === 'user' ? 'flex-start' : 'flex-end',
                      bgcolor: msg.role === 'user' ? '#e3f2fd' : '#e8f5e9',
                      color: 'text.primary',
                      px: 2,
                      py: 1,
                      borderRadius: 2,
                      maxWidth: '80%',
                      boxShadow: 1,
                    }}
                  >
                    <Typography variant="body1" sx={{ whiteSpace: 'pre-line' }}>
                      <ReactMarkdown>{msg.text}</ReactMarkdown>
                    </Typography>
                    {msg.role === 'model' && msg.sources && (
                      <Box sx={{ mt: 1 }}>
                        <Divider sx={{ my: 1 }} />
                        <Typography variant="subtitle2" sx={{ mb: 0.5 }}>Sources:</Typography>
                        {msg.sources.map((source, i) => (
                          <Typography key={i} sx={{ fontSize: 13 }}>
                            {source.file} (Confidence: {(source.confidence * 100).toFixed(2)}%)
                          </Typography>
                        ))}
                      </Box>
                    )}
                  </Box>
                ))}
                <div ref={chatEndRef} />
              </Box>
            </Paper>

            {/* ChatGPT-style input bar */}
            <Box sx={{
              width: '100%',
              maxWidth: 700,
              display: 'flex',
              alignItems: 'center',
              position: 'fixed',
              left: '50%',
              transform: 'translateX(-50%)',
              bottom: 32, // ~3cm (adjust as needed)
              background: '#fff',
              p: 2,
              borderRadius: 2,
              boxShadow: 3,
              zIndex: 10
            }}>
              {/* Dropup for domain selection */}
              <IconButton
                onClick={(e) => setAnchorEl(e.currentTarget)}
                sx={{ mr: 1 }}
                size="large"
              >
                <ArrowDropUpIcon />
              </IconButton>
              <Popover
                open={Boolean(anchorEl)}
                anchorEl={anchorEl}
                onClose={() => setAnchorEl(null)}
                anchorOrigin={{ vertical: 'top', horizontal: 'left' }}
                transformOrigin={{ vertical: 'bottom', horizontal: 'left' }}
              >
                <Box sx={{ p: 2 }}>
                  <Typography variant="subtitle2" sx={{ mb: 1 }}>Select Domain</Typography>
                  <Select
                    value={selectedDomain}
                    onChange={(e) => {
                      setSelectedDomain(e.target.value);
                      setAnchorEl(null);
                    }}
                    fullWidth
                  >
                    {domains.map(domain => (
                      <MenuItem key={domain} value={domain}>{domain}</MenuItem>
                    ))}
                  </Select>
                </Box>
              </Popover>

              {/* Query input */}
              <TextField
                fullWidth
                placeholder="Type your question..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleQuery();
                  }
                }}
                InputProps={{
                  endAdornment: (
                    <InputAdornment position="end">
                      <IconButton
                        color="primary"
                        onClick={handleQuery}
                        disabled={loading || !query.trim()}
                        size="large"
                      >
                        <SendIcon />
                      </IconButton>
                    </InputAdornment>
                  )
                }}
                sx={{ bgcolor: '#f5f5f5', borderRadius: 2 }}
              />
            </Box>
          </Box>
        )}

        {tab === 1 && (
          <Grid container spacing={3} sx={{ maxWidth: 1200, mx: 'auto' }}>
            <Grid item xs={12} md={4}>
              <Paper sx={{ p: 2, maxWidth: 340, width: '100%' }}>
                <Typography variant="h6" gutterBottom>MCQ Parameters</Typography>
                {/* Domain, Topic, Difficulty, etc. */}
                <Select
                  value={selectedDomain}
                  onChange={(e) => setSelectedDomain(e.target.value)}
                  fullWidth
                  sx={{ mb: 2 }}
                >
                  {domains.map(domain => (
                    <MenuItem key={domain} value={domain}>{domain}</MenuItem>
                  ))}
                </Select>
                <TextField
                  fullWidth
                  value={topic}
                  onChange={(e) => setTopic(e.target.value)}
                  label="Topic (optional)"
                  sx={{ mb: 2 }}
                />
                <TextField
                  fullWidth
                  type="number"
                  value={numQuestions}
                  onChange={(e) => setNumQuestions(parseInt(e.target.value))}
                  label="Number of Questions"
                  InputProps={{ inputProps: { min: 1, max: 20 } }}
                  sx={{ mb: 2 }}
                />
                <Select
                  value={difficulty}
                  onChange={(e) => setDifficulty(e.target.value)}
                  fullWidth
                  sx={{ mb: 2 }}
                >
                  {['Easy', 'Medium', 'Hard'].map(level => (
                    <MenuItem key={level} value={level}>{level}</MenuItem>
                  ))}
                </Select>
                <Button
                  variant="contained"
                  onClick={handleMCQ}
                  disabled={loading}
                  fullWidth
                >
                  {loading ? <CircularProgress size={24} /> : 'Generate MCQs'}
                </Button>
              </Paper>
            </Grid>
            <Grid item xs={12} md={8}>
              <Paper sx={{ p: 2, minHeight: 400, minWidth: 0 }}>
                {mcqs ? (
                  <>
                    <Tabs value={mcqView} onChange={(e, newValue) => setMcqView(newValue)}>
                      <Tab label="Student View" />
                      <Tab label="Teacher View" />
                    </Tabs>
                    {mcqView === 0 ? (
                      // Student View
                      <Box sx={{ mt: 2 }}>
                        {mcqs.map((mcq, index) => (
                          <Box key={index} sx={{ mb: 4 }}>
                            <Typography variant="h6">Question {index + 1}:</Typography>
                            <Typography sx={{ mb: 2 }}>{mcq.question}</Typography>
                            {Object.entries(mcq.options).map(([key, value]) => (
                              <Typography key={key} sx={{ ml: 2, mb: 1 }}>
                                {key}) {value}
                              </Typography>
                            ))}
                            <Divider sx={{ my: 2 }} />
                          </Box>
                        ))}
                      </Box>
                    ) : (
                      // Teacher View
                      <Box sx={{ mt: 2 }}>
                        {mcqs.map((mcq, index) => (
                          <Box key={index} sx={{ mb: 4 }}>
                            <Typography variant="h6">Question {index + 1}:</Typography>
                            <Typography sx={{ mb: 2 }}>{mcq.question}</Typography>
                            {Object.entries(mcq.options).map(([key, value]) => (
                              <Typography 
                                key={key} 
                                sx={{ 
                                  ml: 2, 
                                  mb: 1,
                                  fontWeight: mcq.correct_answer === key ? 'bold' : 'normal',
                                  color: mcq.correct_answer === key ? 'success.main' : 'inherit'
                                }}
                              >
                                {key}) {value}
                              </Typography>
                            ))}
                            <Typography sx={{ mt: 1, color: 'info.main' }}>
                              <strong>Correct Answer:</strong> {mcq.correct_answer}
                            </Typography>
                            <Typography sx={{ mt: 1, color: 'text.secondary' }}>
                              <strong>Explanation:</strong> {mcq.explanation}
                            </Typography>
                            <Divider sx={{ my: 2 }} />
                          </Box>
                        ))}
                      </Box>
                    )}
                    <Button
                      variant="outlined"
                      onClick={() => {
                        const jsonStr = JSON.stringify(mcqs, null, 2);
                        const blob = new Blob([jsonStr], { type: 'application/json' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `mcqs_${selectedDomain}_${difficulty}_${numQuestions}.json`;
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        URL.revokeObjectURL(url);
                      }}
                      sx={{ mt: 2 }}
                    >
                      Download MCQs as JSON
                    </Button>
                  </>
                ) : (
                  <Box sx={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <Typography variant="subtitle1" color="text.secondary">
                      Generate MCQs to see them here.
                    </Typography>
                  </Box>
                )}
              </Paper>
            </Grid>
          </Grid>
        )}

        {tab === 2 && (
          <Paper sx={{ p: 3, maxWidth: 700, mx: 'auto' }}>
            <Typography variant="h5" gutterBottom>File Upload</Typography>
            {/* Domain selection */}
            <Autocomplete
              freeSolo
              options={domains}
              value={uploadDomain}
              onChange={(event, newValue) => setUploadDomain(newValue)}
              onInputChange={(event, newInputValue) => setUploadDomain(newInputValue)}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Domain"
                  placeholder="Type or select domain"
                  sx={{ mb: 2 }}
                />
              )}
            />
            {/* Removed Subject input */}
            <input
              type="file"
              multiple
              onChange={(e) => setFiles(Array.from(e.target.files))}
              style={{ marginBottom: 16 }}
            />
            <Button
              variant="contained"
              onClick={handleFileUpload}
              disabled={loading || !uploadDomain || files.length === 0}
            >
              {loading ? <CircularProgress size={24} /> : 'Upload Files'}
            </Button>
          </Paper>
        )}
      </Box>
    </Box>
  );
}

export default App;


