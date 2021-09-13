import React from "react";
import "../styles/Projects.css";
import Cards from "./Cards.js";
import descrip from "../files/descrip";
import ChessML from "../Images/Chess.png";
import BrainANN from "../Images/BrainANN.jpg";

// const yapic = require("yandex-pictures");
// const yandeximages = require("yandex-images");

export default function Project() {
  return (
    <div className="container-fluid project">
      <div className="row ">
        <div className="col-7 ml-4 mt-4">
          <h1 className="h1-project">Projects and Work</h1>
        </div>
        {/* <div className="col-4 pl-5 ml-5 mr-0 pr-0"></div> */}
      </div>
      <div className="container-fluid padding">
        <div className="row">
          <Cards
            link="https://github.com/AliAlShammaa/NeuralNetworks"
            src={BrainANN}
            title={descrip.BrainANN.title}
            txt={descrip.BrainANN.txt}
          />
          <Cards
            link="https://github.com/AliAlShammaa/ChessML"
            src={ChessML}
            title={descrip.Chess.title}
            txt={descrip.Chess.txt}
          />
          <Cards
            link="#"
            src=""
            title={descrip.AA.title}
            txt={descrip.AA.txt}
          />
          <Cards
            link="https://www.atlas365.ca/ecopoints"
            src=""
            title={descrip.Atlas.title}
            txt={descrip.Atlas.txt}
          />
        </div>
      </div>
    </div>
  );
}
