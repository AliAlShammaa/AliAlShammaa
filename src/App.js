import "./App.css";
import React, { useEffect, useState } from "react";
import Home from "./Components/Home.js";
import Header from "./Components/Header.js";
import Social from "./Components/Social.js";
import ANN from "./Components/ANN.js";
import Footer from "./Components/Footer.js";
import { BrowserRouter as Router, Switch, Route } from "react-router-dom";

const gsap = window.gsap;
const tl = gsap.timeline({ defaults: { ease: "power1.out" } });

function App() {
  let bool = false;
  let content = (
    <div>
      <h1 id="siteTitle1" className="hide1">
        <span id="titleName"> Ali Al Shammaa</span>
      </h1>
      <div className="preload">
        <div id="loading" className="flex justify-center">
          <div className="loader"></div>
          <span id="loadingText" className="mt-3">
            Loading...
          </span>
        </div>
      </div>
    </div>
  );

  let content2 = (
    <div className="preload-finnish">
      <div id="loading" className="flex justify-center">
        <div className="loader"></div>
        <span id="loadingText" className="mt-3">
          Loading...
        </span>
      </div>
    </div>
  );

  let [preloader, setPreloader] = useState(content);

  // useEffect(() => {
  //   window.addEventListener("load", () => {
  //     preload.classList.add("preload-finnish");
  //   });
  // }, []);'
  let i = 0;
  useEffect(() => {
    tl.to("#titleName", { y: "20%", duration: 1, stagger: 0.25 });
    setTimeout(() => {
      if (!bool) {
        setPreloader(content2);
        bool = true;
      } else {
        setPreloader(null);
      }
    }, 1250);
  }, [i]);

  // tl.to("#titleName", { y: "20%", duration: 1, stagger: 0.25 });

  return (
    <div id="maindiv" className="container-fluid p-0">
      {preloader}
      <div>
        <section className="section">
          <Router basename={process.env.PUBLIC_URL}>
            <Header />
            <Switch>
              <Route path="/" exact component={Home} />
              {i++}
              <Route path="/ANNandIntelligenceGuide" exact component={ANN} />
            </Switch>
          </Router>
          <Social />
        </section>
        <Footer />
      </div>
    </div>
  );
}

export default App;
