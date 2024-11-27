const choices = document.querySelectorAll('.choice');
const message = document.getElementById('message');
const result = document.getElementById('result');
const playerImg = document.getElementById('player-img');
const computerImg = document.getElementById('computer-img');
const totalGamesElement = document.getElementById('total-games');
const totalWinsElement = document.getElementById('total-wins');
const totalLoosesElement = document.getElementById('total-looses');
const totalDrawElement = document.getElementById('total-draw');


const questionImage = 'Pictures/question.png';

const images = {
    rock: 'Pictures/rock.png',
    paper: 'Pictures/paper.png',
    scissors: 'Pictures/scissors.png',
    bottle: 'Pictures/bottle.png',
    question: questionImage
};

// Game counters
let totalGames = 0;
let totalWins = 0;
let totalLooses = 0;
let totalDraw = 0;


// Set initial images
playerImg.src = images.question;
computerImg.src = images.question;

choices.forEach(choice => {
    choice.addEventListener('click', function() {
        const playerChoice = this.id;
        const computerChoice = gameOptions[Math.floor(Math.random() * 3)];
        const outcome = getResult(playerChoice, computerChoice);

        // Update message and result
        message.textContent = `You chose ${playerChoice}, computer chose ${computerChoice}.`;
        result.textContent = outcome;

        // Show the selected images for player and computer
        playerImg.src = images[playerChoice];
        computerImg.src = images[computerChoice];

        // Update counters
        totalGames++;
        if (outcome.includes("win")) {
            totalWins++;
        } else if (outcome.includes("lose")) {
            totalLooses++;
        } else {
            totalDraw++
        } 
    
        
        totalGamesElement.textContent = totalGames;
        totalWinsElement.textContent = totalWins;
        totalLoosesElement.textContent = totalLooses;
        totalDrawElement.textContent = totalDraw;
        
        // Animate the result fade-in
        result.style.opacity = '0'; // Reset result opacity for fade-in
        setTimeout(() => {
            result.style.opacity = '1'; // Animate result fade-in
        }, 300);
        
        // Update result color
        result.className = ''; // Reset previous classes
        if (outcome.includes("win")) {
            result.classList.add('result-win');
        } else if (outcome.includes("lose")) {
            result.classList.add('result-lose');
        } else {
            result.classList.add('result-draw');
        }
    });
});

function getResult(player, computer) {
    if (player === computer) {
        return "It's a draw!";
    } else if (
        (player === 'rock' && computer === 'scissors') ||
        (player === 'scissors' && computer === 'paper') ||
        (player === 'paper' && computer === 'rock') ||
        (player === 'bottle' && computer === 'rock') ||
        (player === 'bottle' && computer === 'paper') || 
        (player === 'rock' && computer === 'bottle')
    ) {
        return "You win!";
    } else {
        return "You lose!";
    }
}

const gameOptions = ['rock', 'paper', 'scissors' , 'bottle'];
