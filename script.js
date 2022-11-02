//Random message generator
//F1 drivers and accions in spanish

//F1 names and last names

//F1 circuits

//Time in seconds of sector of track

//Adjetives for the time

//SAMPLE MESSAGE
// Lewis Hamilton marco un tiempo de 42.245seg en el sector 5 del circuito de Monza
// ${name} marco un tiempo de ${tiempoSeg}seg en el sector ${sectorNumber} del circuito de ${nameCircuit}
const namePilotsArray = [
  "Yuki Tsunoda",
  "Lando Norris",
  "Zhou Guanyu",
  "Mick Schumacher",
  "Lance Stroll",
  "George Russell",
  "Charles Leclerc",
  "Max Verstappen",
  "Esteban Ocon",
  "Alex Albon",
  "Pierre Gasly",
  "Nicholas Latifi",
  "Carlos Sainz",
  "Kevin Magnussen",
  "Valtteri Bottas",
  "Daniel Ricciardo",
  "Sergio Perez",
  "Sebastian Vettel",
  "Lewis Hamilton",
  "Fernando Alonso",
];

const sectorTime = () => (Math.random() * 120).toFixed(3);

const sectorName = () => Math.floor(Math.random() * 30);

const nameCircuitsArray = [
  "A1-Ring",
  "Adelaide Street Circuit",
  "Ain-Diab",
  "Aintree",
  "Albert Park",
  "Algarve International Circuit",
  "AVUS",
  "Bahrain International Circuit",
  "Baku City Circuit",
  "Circuit de Barcelona-Catalunya",
  "Circuito da Boavista",
  "Brands Hatch",
  "Circuit Bremgarten",
  "Buddh International Circuit",
  "Bugatti Circuit",
  "Caesars Palace",
  "Charade Circuit",
  "Circuit of the Americas",
  "Detroit Street Circuit",
  "Dijon-Prenois",
  "Donington Park",
  "Autodromo Enzo e Dino Ferrari",
  "Autódromo do Estoril",
  "Fair Park circuit",
  "Fuji Speedway",
  "Circuit Gilles Villeneuve",
  "Hanoi Street Circuit",
  "Autódromo Hermanos Rodríguez",
  "Hockenheimring",
  "Hungaroring",
  "Igora Drive",
  "Indianapolis Motor Speedway",
  "Istanbul Park",
  "Autódromo Internacional Nelson Piquet",
  "Circuito del Jarama",
  "Jeddah Street Circuit",
  "Circuito de Jerez",
  "Autódromo José Carlos Pace",
  "Korea International Circuit",
  "Kyalami Circuit",
  "Long Beach street circuit",
  "Losail International Circuit",
  "Circuit de Nevers Magny-Cours",
  "Marina Bay Street Circuit",
  "Miami International Autodrome",
  "Circuit de Monaco",
  "Circuito de Monsanto",
  "Montjuïc Park Circuit",
  "Circuit Mont-Tremblant",
  "Autodromo Nazionale Monza",
  "Mosport Park",
  "Mugello Circuit",
  "Nivelles-Baulers",
  "Nürburgring",
  "Okayama International Circuit",
  "Autódromo Juan y Oscar Gálvez",
  "Österreichring",
  "Circuit Paul Ricard",
  "Pedralbes Circuit",
  "Pescara Circuit",
  "Phoenix Street Circuit",
  "Prince George Circuit",
  "Red Bull Ring",
  "Reims-Gueux",
  "Riverside International Raceway",
  "Rouen-Les-Essarts",
  "Scandinavian Raceway",
  "Sebring International Raceway",
  "Sepang International Circuit",
  "Shanghai International Circuit",
  "Silverstone Circuit",
  "Sochi Autodrom",
  "Circuit de Spa-Francorchamps",
  "Suzuka Circuit",
  "Valencia Street Circuit",
  "Watkins Glen International",
  "Yas Marina Circuit",
  "Circuit Park Zandvoort",
  "Zeltweg Airfield",
  "Circuit Zolder",
];

const returnRandomString = array => array[Math.floor(Math.random()*array.length)]

//console.log(returnRandomString(namePilotsArray))

const randonMessage = () => `${returnRandomString(namePilotsArray)} marcó un tiempo de ${sectorTime()}seg. en el sector Nº ${sectorName()} del circuito de ${returnRandomString(nameCircuitsArray)}`

console.log(randonMessage())