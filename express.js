const express = require('express');
const bodyParser = require('body-parser');
const { spawn } = require('child_process');
const cors = require('cors');
const multer = require('multer');
const upload = multer({ dest: 'uploads/' });
const up = multer({ dest: 'liverecording/' });
const PORT = 5000;

const app = express();
app.use(cors());
app.use(bodyParser.json());



app.post('/analyze-text', (req, res) => {
    const text = req.body.text;
    const pythonProcess = spawn('python', ['ai.py', text]);
    let result = '';
    
    pythonProcess.stdout.on('data', (data) => {
        result += data.toString();
    });

    pythonProcess.on('close', (code) => {
        if (code === 0) {
            res.json({ result: result.trim() });
        } else {
            res.status(500).json({ error: 'Error executing Python script' });
        }
    });

    
});
//orginal audio

app.post('/analyze-audio', upload.single('audio'), (req, res) => {
    const audioPath = req.file.path;
    
    console.log("Received audio file:", audioPath);  
   
    const pythonProcess = spawn('python', ['Live.py', audioPath]);
    let result = '';

    pythonProcess.stdout.on('data', (data) => {
       
        console.log("Python script output:", data.toString()); 
       
        result += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
    });

    pythonProcess.on('close', (code) => {
       
        console.log(`Python script exited with code ${code}`); 
       
        if (code === 0) {
            res.json({ result: result.trim() });
        } else {
            res.status(500).json({ error: 'Error executing Python script' });
        }
    });
});


app.post("/upload", up.single("audio"), (req, res) => {
    const filePath = req.file.path;
    console.log("Received audio file:", filePath);  // Log the path of the received audio file

    const pythonProcess = spawn("python", ["Live.py", filePath]);
    pythonProcess.stdout.on("data", (data) => {
        const result = data.toString();
        console.log("Python script result:", result);  // Log the result from the Python script
        res.json({ result: result.trim() });
    });
    pythonProcess.stderr.on("data", (data) => {
     //   console.error("Python script error:", data.toString()); 
    });
});

app.listen(PORT, () => console.log(`Server is running on port ${PORT}`));


