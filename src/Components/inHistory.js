import { React, useState, useEffect } from "react";
import Carousel from "./Carousel";
import "../styles/inHistory.css";

export default function History() {
  var today = new Date();
  var dd = String(today.getDate());
  var mm = String(today.getMonth() + 1);
  let date = null;
  let proxyUrl = "https://cors-anywhere.herokuapp.com/";
  let targetUrl = `https://history.muffinlabs.com/date/${mm}/${dd}`;
  let targetUrl2 =
    "https://www.google.com/search?q=google+images&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjzoui4qfbtAhUqFFkFHTHRAG8Q_AUoAXoECBIQAw&biw=1600&bih=722";
  let storage = window.sessionStorage;
  let i = 0;

  let updateCount = function () {
    if (!storage.getItem("i")) {
      storage.setItem("i", i);
      console.log("hel");
    } else {
      i = parseInt(storage.getItem("i")) + 3;
      storage.setItem("i", i);
      console.log("hey ");
    }
    fetchEvents();
  };

  useEffect(() => {
    fetchEvents();
  }, []);

  let [events, setEvents] = useState([]);
  // let eventsArray = [];
  // let setEventsArray = function (x) {
  //   eventsArray = x;
  // };

  const fetchEvents = async () => {
    const data = await fetch(proxyUrl + targetUrl);
    // const images = await fetch(proxyUrl + targetUrl2);
    // console.log(images);
    const object = await data.json();
    date = object.date;
    date = date.substr(0, 3).concat(date.substr(date.length - 2));
    console.log(i, date, object);
    let eventsArray = object.data.Events;
    console.log(eventsArray);
    if (eventsArray.length === 0) {
      events = new Array(3).fill("").map((x) => {
        return new Array(2).fill("");
      });
    } else if (eventsArray.length > 0) {
      console.log(eventsArray);
      events = [
        [eventsArray[i].text, eventsArray[i].year],
        [eventsArray[i + 1].text, eventsArray[i + 1].year],
        [eventsArray[i + 2].text, eventsArray[i + 2].year],
        date,
      ];
      console.log(events);
    }
    console.log(events);
    setEvents(events);
  };

  if (events.length === 0) {
    events = new Array(3).fill("").map((x) => {
      return new Array(2).fill("");
    });
  }

  // function setEvents() {

  // }

  return (
    <div className="hist container-fluid padding">
      <h1>Today in history</h1>
      <h3>
        {" "}
        While you're here, enjoy some history 🧐😁. This feature is made with
        API from https://history.muffinlabs.com.
      </h3>
      <button
        className="btn btn-outline-info btn-dark mt-2 p-3"
        onClick={updateCount}
      >
        <i className="fa fa-refresh " /> See More!
      </button>
      <Carousel
        date={events[3]}
        txt1={events[0][0]}
        img1=""
        yr1={events[0][1]}
        txt2={events[1][0]}
        yr2={events[1][1]}
        txt3={events[2][0]}
        yr3={events[2][1]}
      />
    </div>
  );
}
