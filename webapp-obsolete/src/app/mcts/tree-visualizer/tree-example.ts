export let example = {
  "state": [[0, 1, 0], [0, -1, 0], [0, 0, 0]],
  "actionTaken": null,
  "value": 6,
  "visits": 100,
  "expandableMoves": [0, 0, 0, 0, 0, 0, 0, 0, 0],
  "children": [
    {
      "value": -2,
      "actionTaken": 8,
      "visits": 15,
      "expandableMoves": [0, 0, 0, 0, 0, 0, 0, 0, 0],
      "children": [
        {
          "value": 2,
          "actionTaken": 7,
          "visits": 2,
          "expandableMoves": [1, 0, 0, 1, 0, 1, 1, 0, 0],
          "children": [
            {
              "value": -1,
              "actionTaken": 2,
              "visits": 1,
              "expandableMoves": [1, 0, 0, 1, 0, 1, 1, 0, 0],
              "children": [],
              "state": [[0, -1, -1], [0, 1, 0], [0, 1, -1]]
            }
          ],
          "state": [[0, 1, 0], [0, -1, 0], [0, -1, 1]]
        },
        {
          "value": -1,
          "actionTaken": 6,
          "visits": 3,
          "expandableMoves": [1, 0, 0, 1, 0, 0, 0, 1, 0],
          "children": [
            {
              "value": 1,
              "actionTaken": 5,
              "visits": 1,
              "expandableMoves": [1, 0, 1, 1, 0, 0, 0, 1, 0],
              "children": [],
              "state": [[0, -1, 0], [0, 1, -1], [1, 0, -1]]
            },
            {
              "value": -1,
              "actionTaken": 2,
              "visits": 1,
              "expandableMoves": [1, 0, 0, 1, 0, 1, 0, 1, 0],
              "children": [],
              "state": [[0, -1, -1], [0, 1, 0], [1, 0, -1]]
            }
          ],
          "state": [[0, 1, 0], [0, -1, 0], [-1, 0, 1]]
        },
        {
          "value": 1,
          "actionTaken": 0,
          "visits": 2,
          "expandableMoves": [0, 0, 1, 1, 0, 1, 0, 1, 0],
          "children": [
            {
              "value": 0,
              "actionTaken": 6,
              "visits": 1,
              "expandableMoves": [0, 0, 1, 1, 0, 1, 0, 1, 0],
              "children": [],
              "state": [[1, -1, 0], [0, 1, 0], [-1, 0, -1]]
            }
          ],
          "state": [[-1, 1, 0], [0, -1, 0], [0, 0, 1]]
        },
        {
          "value": -2,
          "actionTaken": 3,
          "visits": 4,
          "expandableMoves": [1, 0, 1, 0, 0, 0, 0, 0, 0],
          "children": [
            {
              "value": 1,
              "actionTaken": 7,
              "visits": 1,
              "expandableMoves": [1, 0, 1, 0, 0, 1, 1, 0, 0],
              "children": [],
              "state": [[0, -1, 0], [1, 1, 0], [0, -1, -1]]
            },
            {
              "value": 1,
              "actionTaken": 6,
              "visits": 1,
              "expandableMoves": [1, 0, 1, 0, 0, 1, 0, 1, 0],
              "children": [],
              "state": [[0, -1, 0], [1, 1, 0], [-1, 0, -1]]
            },
            {
              "value": -1,
              "actionTaken": 5,
              "visits": 1,
              "expandableMoves": [1, 0, 1, 0, 0, 0, 1, 1, 0],
              "children": [],
              "state": [[0, -1, 0], [1, 1, -1], [0, 0, -1]]
            }
          ],
          "state": [[0, 1, 0], [-1, -1, 0], [0, 0, 1]]
        },
        {
          "value": 0,
          "actionTaken": 5,
          "visits": 2,
          "expandableMoves": [0, 0, 1, 1, 0, 0, 1, 1, 0],
          "children": [
            {
              "value": -1,
              "actionTaken": 0,
              "visits": 1,
              "expandableMoves": [0, 0, 1, 1, 0, 0, 1, 1, 0],
              "children": [],
              "state": [[-1, -1, 0], [0, 1, 1], [0, 0, -1]]
            }
          ],
          "state": [[0, 1, 0], [0, -1, -1], [0, 0, 1]]
        },
        {
          "value": 1,
          "actionTaken": 2,
          "visits": 1,
          "expandableMoves": [1, 0, 0, 1, 0, 1, 1, 1, 0],
          "children": [],
          "state": [[0, 1, -1], [0, -1, 0], [0, 0, 1]]
        }
      ],
      "state": [[0, -1, 0], [0, 1, 0], [0, 0, -1]]
    },
    {
      "value": 2,
      "actionTaken": 5,
      "visits": 12,
      "expandableMoves": [0, 0, 0, 0, 0, 0, 0, 0, 0],
      "children": [
        {
          "value": 1,
          "actionTaken": 3,
          "visits": 1,
          "expandableMoves": [1, 0, 1, 0, 0, 0, 1, 1, 1],
          "children": [],
          "state": [[0, 1, 0], [-1, -1, 1], [0, 0, 0]]
        },
        {
          "value": -3,
          "actionTaken": 6,
          "visits": 3,
          "expandableMoves": [0, 0, 1, 0, 0, 0, 0, 1, 1],
          "children": [
            {
              "value": 1,
              "actionTaken": 3,
              "visits": 1,
              "expandableMoves": [1, 0, 1, 0, 0, 0, 0, 1, 1],
              "children": [],
              "state": [[0, -1, 0], [-1, 1, -1], [1, 0, 0]]
            },
            {
              "value": 1,
              "actionTaken": 0,
              "visits": 1,
              "expandableMoves": [0, 0, 1, 1, 0, 0, 0, 1, 1],
              "children": [],
              "state": [[-1, -1, 0], [0, 1, -1], [1, 0, 0]]
            }
          ],
          "state": [[0, 1, 0], [0, -1, 1], [-1, 0, 0]]
        },
        {
          "value": 1,
          "actionTaken": 2,
          "visits": 2,
          "expandableMoves": [0, 0, 0, 1, 0, 0, 1, 1, 1],
          "children": [
            {
              "value": -1,
              "actionTaken": 0,
              "visits": 1,
              "expandableMoves": [0, 0, 0, 1, 0, 0, 1, 1, 1],
              "children": [],
              "state": [[-1, -1, 1], [0, 1, -1], [0, 0, 0]]
            }
          ],
          "state": [[0, 1, -1], [0, -1, 1], [0, 0, 0]]
        },
        {
          "value": -1,
          "actionTaken": 7,
          "visits": 2,
          "expandableMoves": [1, 0, 1, 0, 0, 0, 1, 0, 1],
          "children": [
            {
              "value": 1,
              "actionTaken": 3,
              "visits": 1,
              "expandableMoves": [1, 0, 1, 0, 0, 0, 1, 0, 1],
              "children": [],
              "state": [[0, -1, 0], [-1, 1, -1], [0, 1, 0]]
            }
          ],
          "state": [[0, 1, 0], [0, -1, 1], [0, -1, 0]]
        },
        {
          "value": 1,
          "actionTaken": 0,
          "visits": 1,
          "expandableMoves": [0, 0, 1, 1, 0, 0, 1, 1, 1],
          "children": [],
          "state": [[-1, 1, 0], [0, -1, 1], [0, 0, 0]]
        },
        {
          "value": 0,
          "actionTaken": 8,
          "visits": 2,
          "expandableMoves": [1, 0, 0, 1, 0, 0, 1, 1, 0],
          "children": [
            {
              "value": -1,
              "actionTaken": 2,
              "visits": 1,
              "expandableMoves": [1, 0, 0, 1, 0, 0, 1, 1, 0],
              "children": [],
              "state": [[0, -1, -1], [0, 1, -1], [0, 0, 1]]
            }
          ],
          "state": [[0, 1, 0], [0, -1, 1], [0, 0, -1]]
        }
      ],
      "state": [[0, -1, 0], [0, 1, -1], [0, 0, 0]]
    },
    {
      "value": -2,
      "actionTaken": 0,
      "visits": 15,
      "expandableMoves": [0, 0, 0, 0, 0, 0, 0, 0, 0],
      "children": [
        {
          "value": 0,
          "actionTaken": 5,
          "visits": 2,
          "expandableMoves": [0, 0, 1, 1, 0, 0, 1, 0, 1],
          "children": [
            {
              "value": 1,
              "actionTaken": 7,
              "visits": 1,
              "expandableMoves": [0, 0, 1, 1, 0, 0, 1, 0, 1],
              "children": [],
              "state": [[-1, -1, 0], [0, 1, 1], [0, -1, 0]]
            }
          ],
          "state": [[1, 1, 0], [0, -1, -1], [0, 0, 0]]
        },
        {
          "value": 2,
          "actionTaken": 7,
          "visits": 2,
          "expandableMoves": [0, 0, 1, 1, 0, 1, 1, 0, 0],
          "children": [
            {
              "value": -1,
              "actionTaken": 8,
              "visits": 1,
              "expandableMoves": [0, 0, 1, 1, 0, 1, 1, 0, 0],
              "children": [],
              "state": [[-1, -1, 0], [0, 1, 0], [0, 1, -1]]
            }
          ],
          "state": [[1, 1, 0], [0, -1, 0], [0, -1, 0]]
        },
        {
          "value": 0,
          "actionTaken": 6,
          "visits": 2,
          "expandableMoves": [0, 0, 1, 1, 0, 0, 0, 1, 1],
          "children": [
            {
              "value": -1,
              "actionTaken": 5,
              "visits": 1,
              "expandableMoves": [0, 0, 1, 1, 0, 0, 0, 1, 1],
              "children": [],
              "state": [[-1, -1, 0], [0, 1, -1], [1, 0, 0]]
            }
          ],
          "state": [[1, 1, 0], [0, -1, 0], [-1, 0, 0]]
        },
        {
          "value": -3,
          "actionTaken": 3,
          "visits": 4,
          "expandableMoves": [0, 0, 1, 0, 0, 0, 0, 0, 1],
          "children": [
            {
              "value": 1,
              "actionTaken": 5,
              "visits": 1,
              "expandableMoves": [0, 0, 1, 0, 0, 0, 1, 1, 1],
              "children": [],
              "state": [[-1, -1, 0], [1, 1, -1], [0, 0, 0]]
            },
            {
              "value": 0,
              "actionTaken": 6,
              "visits": 1,
              "expandableMoves": [0, 0, 1, 0, 0, 1, 0, 1, 1],
              "children": [],
              "state": [[-1, -1, 0], [1, 1, 0], [-1, 0, 0]]
            },
            {
              "value": 1,
              "actionTaken": 7,
              "visits": 1,
              "expandableMoves": [0, 0, 1, 0, 0, 1, 1, 0, 1],
              "children": [],
              "state": [[-1, -1, 0], [1, 1, 0], [0, -1, 0]]
            }
          ],
          "state": [[1, 1, 0], [-1, -1, 0], [0, 0, 0]]
        },
        {
          "value": 2,
          "actionTaken": 8,
          "visits": 2,
          "expandableMoves": [0, 0, 1, 1, 0, 0, 1, 1, 0],
          "children": [
            {
              "value": -1,
              "actionTaken": 5,
              "visits": 1,
              "expandableMoves": [0, 0, 1, 1, 0, 0, 1, 1, 0],
              "children": [],
              "state": [[-1, -1, 0], [0, 1, -1], [0, 0, 1]]
            }
          ],
          "state": [[1, 1, 0], [0, -1, 0], [0, 0, -1]]
        },
        {
          "value": 0,
          "actionTaken": 2,
          "visits": 2,
          "expandableMoves": [0, 0, 0, 0, 0, 1, 1, 1, 1],
          "children": [
            {
              "value": 1,
              "actionTaken": 3,
              "visits": 1,
              "expandableMoves": [0, 0, 0, 0, 0, 1, 1, 1, 1],
              "children": [],
              "state": [[-1, -1, 1], [-1, 1, 0], [0, 0, 0]]
            }
          ],
          "state": [[1, 1, -1], [0, -1, 0], [0, 0, 0]]
        }
      ],
      "state": [[-1, -1, 0], [0, 1, 0], [0, 0, 0]]
    },
    {
      "value": -5,
      "actionTaken": 2,
      "visits": 19,
      "expandableMoves": [0, 0, 0, 0, 0, 0, 0, 0, 0],
      "children": [
        {
          "value": 1,
          "actionTaken": 3,
          "visits": 3,
          "expandableMoves": [0, 0, 0, 0, 0, 1, 1, 0, 1],
          "children": [
            {
              "value": 1,
              "actionTaken": 7,
              "visits": 1,
              "expandableMoves": [1, 0, 0, 0, 0, 1, 1, 0, 1],
              "children": [],
              "state": [[0, -1, -1], [1, 1, 0], [0, -1, 0]]
            },
            {
              "value": -1,
              "actionTaken": 0,
              "visits": 1,
              "expandableMoves": [0, 0, 0, 0, 0, 1, 1, 1, 1],
              "children": [],
              "state": [[-1, -1, -1], [1, 1, 0], [0, 0, 0]]
            }
          ],
          "state": [[0, 1, 1], [-1, -1, 0], [0, 0, 0]]
        },
        {
          "value": -1,
          "actionTaken": 0,
          "visits": 4,
          "expandableMoves": [0, 0, 0, 0, 0, 0, 0, 1, 1],
          "children": [
            {
              "value": 1,
              "actionTaken": 6,
              "visits": 1,
              "expandableMoves": [0, 0, 0, 1, 0, 1, 0, 1, 1],
              "children": [],
              "state": [[1, -1, -1], [0, 1, 0], [-1, 0, 0]]
            },
            {
              "value": 1,
              "actionTaken": 3,
              "visits": 1,
              "expandableMoves": [0, 0, 0, 0, 0, 1, 1, 1, 1],
              "children": [],
              "state": [[1, -1, -1], [-1, 1, 0], [0, 0, 0]]
            },
            {
              "value": -1,
              "actionTaken": 5,
              "visits": 1,
              "expandableMoves": [0, 0, 0, 1, 0, 0, 1, 1, 1],
              "children": [],
              "state": [[1, -1, -1], [0, 1, -1], [0, 0, 0]]
            }
          ],
          "state": [[-1, 1, 1], [0, -1, 0], [0, 0, 0]]
        },
        {
          "value": -1,
          "actionTaken": 5,
          "visits": 4,
          "expandableMoves": [0, 0, 0, 0, 0, 0, 1, 0, 1],
          "children": [
            {
              "value": -1,
              "actionTaken": 0,
              "visits": 1,
              "expandableMoves": [0, 0, 0, 1, 0, 0, 1, 1, 1],
              "children": [],
              "state": [[-1, -1, -1], [0, 1, 1], [0, 0, 0]]
            },
            {
              "value": 1,
              "actionTaken": 3,
              "visits": 1,
              "expandableMoves": [1, 0, 0, 0, 0, 0, 1, 1, 1],
              "children": [],
              "state": [[0, -1, -1], [-1, 1, 1], [0, 0, 0]]
            },
            {
              "value": 1,
              "actionTaken": 7,
              "visits": 1,
              "expandableMoves": [1, 0, 0, 1, 0, 0, 1, 0, 1],
              "children": [],
              "state": [[0, -1, -1], [0, 1, 1], [0, -1, 0]]
            }
          ],
          "state": [[0, 1, 1], [0, -1, -1], [0, 0, 0]]
        },
        {
          "value": 2,
          "actionTaken": 8,
          "visits": 2,
          "expandableMoves": [1, 0, 0, 1, 0, 0, 1, 1, 0],
          "children": [
            {
              "value": -1,
              "actionTaken": 5,
              "visits": 1,
              "expandableMoves": [1, 0, 0, 1, 0, 0, 1, 1, 0],
              "children": [],
              "state": [[0, -1, -1], [0, 1, -1], [0, 0, 1]]
            }
          ],
          "state": [[0, 1, 1], [0, -1, 0], [0, 0, -1]]
        },
        {
          "value": 2,
          "actionTaken": 7,
          "visits": 2,
          "expandableMoves": [1, 0, 0, 1, 0, 0, 1, 0, 1],
          "children": [
            {
              "value": -1,
              "actionTaken": 5,
              "visits": 1,
              "expandableMoves": [1, 0, 0, 1, 0, 0, 1, 0, 1],
              "children": [],
              "state": [[0, -1, -1], [0, 1, -1], [0, 1, 0]]
            }
          ],
          "state": [[0, 1, 1], [0, -1, 0], [0, -1, 0]]
        },
        {
          "value": 1,
          "actionTaken": 6,
          "visits": 3,
          "expandableMoves": [1, 0, 0, 0, 0, 1, 0, 0, 1],
          "children": [
            {
              "value": 1,
              "actionTaken": 7,
              "visits": 1,
              "expandableMoves": [1, 0, 0, 1, 0, 1, 0, 0, 1],
              "children": [],
              "state": [[0, -1, -1], [0, 1, 0], [1, -1, 0]]
            },
            {
              "value": -1,
              "actionTaken": 3,
              "visits": 1,
              "expandableMoves": [1, 0, 0, 0, 0, 1, 0, 1, 1],
              "children": [],
              "state": [[0, -1, -1], [-1, 1, 0], [1, 0, 0]]
            }
          ],
          "state": [[0, 1, 1], [0, -1, 0], [-1, 0, 0]]
        }
      ],
      "state": [[0, -1, -1], [0, 1, 0], [0, 0, 0]]
    },
    {
      "value": 5,
      "actionTaken": 7,
      "visits": 7,
      "expandableMoves": [0, 0, 0, 0, 0, 0, 0, 0, 0],
      "children": [
        {
          "value": -1,
          "actionTaken": 6,
          "visits": 1,
          "expandableMoves": [1, 0, 1, 1, 0, 1, 0, 0, 1],
          "children": [],
          "state": [[0, 1, 0], [0, -1, 0], [-1, 1, 0]]
        },
        {
          "value": 1,
          "actionTaken": 2,
          "visits": 1,
          "expandableMoves": [1, 0, 0, 1, 0, 1, 1, 0, 1],
          "children": [],
          "state": [[0, 1, -1], [0, -1, 0], [0, 1, 0]]
        },
        {
          "value": -1,
          "actionTaken": 5,
          "visits": 1,
          "expandableMoves": [1, 0, 1, 1, 0, 0, 1, 0, 1],
          "children": [],
          "state": [[0, 1, 0], [0, -1, -1], [0, 1, 0]]
        },
        {
          "value": -1,
          "actionTaken": 0,
          "visits": 1,
          "expandableMoves": [0, 0, 1, 1, 0, 1, 1, 0, 1],
          "children": [],
          "state": [[-1, 1, 0], [0, -1, 0], [0, 1, 0]]
        },
        {
          "value": -1,
          "actionTaken": 3,
          "visits": 1,
          "expandableMoves": [1, 0, 1, 0, 0, 1, 1, 0, 1],
          "children": [],
          "state": [[0, 1, 0], [-1, -1, 0], [0, 1, 0]]
        },
        {
          "value": -1,
          "actionTaken": 8,
          "visits": 1,
          "expandableMoves": [1, 0, 1, 1, 0, 1, 1, 0, 0],
          "children": [],
          "state": [[0, 1, 0], [0, -1, 0], [0, 1, -1]]
        }
      ],
      "state": [[0, -1, 0], [0, 1, 0], [0, -1, 0]]
    },
    {
      "value": -4,
      "actionTaken": 3,
      "visits": 18,
      "expandableMoves": [0, 0, 0, 0, 0, 0, 0, 0, 0],
      "children": [
        {
          "value": 1,
          "actionTaken": 6,
          "visits": 3,
          "expandableMoves": [1, 0, 0, 0, 0, 0, 0, 1, 1],
          "children": [
            {
              "value": 1,
              "actionTaken": 5,
              "visits": 1,
              "expandableMoves": [1, 0, 1, 0, 0, 0, 0, 1, 1],
              "children": [],
              "state": [[0, -1, 0], [-1, 1, -1], [1, 0, 0]]
            },
            {
              "value": -1,
              "actionTaken": 2,
              "visits": 1,
              "expandableMoves": [1, 0, 0, 0, 0, 1, 0, 1, 1],
              "children": [],
              "state": [[0, -1, -1], [-1, 1, 0], [1, 0, 0]]
            }
          ],
          "state": [[0, 1, 0], [1, -1, 0], [-1, 0, 0]]
        },
        {
          "value": -1,
          "actionTaken": 2,
          "visits": 4,
          "expandableMoves": [0, 0, 0, 0, 0, 1, 0, 1, 0],
          "children": [
            {
              "value": 1,
              "actionTaken": 6,
              "visits": 1,
              "expandableMoves": [1, 0, 0, 0, 0, 1, 0, 1, 1],
              "children": [],
              "state": [[0, -1, 1], [-1, 1, 0], [-1, 0, 0]]
            },
            {
              "value": -1,
              "actionTaken": 0,
              "visits": 1,
              "expandableMoves": [0, 0, 0, 0, 0, 1, 1, 1, 1],
              "children": [],
              "state": [[-1, -1, 1], [-1, 1, 0], [0, 0, 0]]
            },
            {
              "value": 0,
              "actionTaken": 8,
              "visits": 1,
              "expandableMoves": [1, 0, 0, 0, 0, 1, 1, 1, 0],
              "children": [],
              "state": [[0, -1, 1], [-1, 1, 0], [0, 0, -1]]
            }
          ],
          "state": [[0, 1, -1], [1, -1, 0], [0, 0, 0]]
        },
        {
          "value": 1,
          "actionTaken": 7,
          "visits": 3,
          "expandableMoves": [0, 0, 1, 0, 0, 0, 1, 0, 1],
          "children": [
            {
              "value": 1,
              "actionTaken": 5,
              "visits": 1,
              "expandableMoves": [1, 0, 1, 0, 0, 0, 1, 0, 1],
              "children": [],
              "state": [[0, -1, 0], [-1, 1, -1], [0, 1, 0]]
            },
            {
              "value": -1,
              "actionTaken": 0,
              "visits": 1,
              "expandableMoves": [0, 0, 1, 0, 0, 1, 1, 0, 1],
              "children": [],
              "state": [[-1, -1, 0], [-1, 1, 0], [0, 1, 0]]
            }
          ],
          "state": [[0, 1, 0], [1, -1, 0], [0, -1, 0]]
        },
        {
          "value": 0,
          "actionTaken": 0,
          "visits": 3,
          "expandableMoves": [0, 0, 0, 0, 0, 0, 1, 1, 1],
          "children": [
            {
              "value": 0,
              "actionTaken": 5,
              "visits": 1,
              "expandableMoves": [0, 0, 1, 0, 0, 0, 1, 1, 1],
              "children": [],
              "state": [[1, -1, 0], [-1, 1, -1], [0, 0, 0]]
            },
            {
              "value": 1,
              "actionTaken": 2,
              "visits": 1,
              "expandableMoves": [0, 0, 0, 0, 0, 1, 1, 1, 1],
              "children": [],
              "state": [[1, -1, -1], [-1, 1, 0], [0, 0, 0]]
            }
          ],
          "state": [[-1, 1, 0], [1, -1, 0], [0, 0, 0]]
        },
        {
          "value": 2,
          "actionTaken": 8,
          "visits": 2,
          "expandableMoves": [0, 0, 1, 0, 0, 1, 1, 1, 0],
          "children": [
            {
              "value": -1,
              "actionTaken": 0,
              "visits": 1,
              "expandableMoves": [0, 0, 1, 0, 0, 1, 1, 1, 0],
              "children": [],
              "state": [[-1, -1, 0], [-1, 1, 0], [0, 0, 1]]
            }
          ],
          "state": [[0, 1, 0], [1, -1, 0], [0, 0, -1]]
        },
        {
          "value": 2,
          "actionTaken": 5,
          "visits": 2,
          "expandableMoves": [1, 0, 1, 0, 0, 0, 0, 1, 1],
          "children": [
            {
              "value": -1,
              "actionTaken": 6,
              "visits": 1,
              "expandableMoves": [1, 0, 1, 0, 0, 0, 0, 1, 1],
              "children": [],
              "state": [[0, -1, 0], [-1, 1, 1], [-1, 0, 0]]
            }
          ],
          "state": [[0, 1, 0], [1, -1, -1], [0, 0, 0]]
        }
      ],
      "state": [[0, -1, 0], [-1, 1, 0], [0, 0, 0]]
    },
    {
      "value": 0,
      "actionTaken": 6,
      "visits": 14,
      "expandableMoves": [0, 0, 0, 0, 0, 0, 0, 0, 0],
      "children": [
        {
          "value": 1,
          "actionTaken": 2,
          "visits": 1,
          "expandableMoves": [1, 0, 0, 1, 0, 1, 0, 1, 1],
          "children": [],
          "state": [[0, 1, -1], [0, -1, 0], [1, 0, 0]]
        },
        {
          "value": 0,
          "actionTaken": 8,
          "visits": 2,
          "expandableMoves": [1, 0, 0, 1, 0, 1, 0, 1, 0],
          "children": [
            {
              "value": -1,
              "actionTaken": 2,
              "visits": 1,
              "expandableMoves": [1, 0, 0, 1, 0, 1, 0, 1, 0],
              "children": [],
              "state": [[0, -1, -1], [0, 1, 0], [-1, 0, 1]]
            }
          ],
          "state": [[0, 1, 0], [0, -1, 0], [1, 0, -1]]
        },
        {
          "value": -2,
          "actionTaken": 5,
          "visits": 3,
          "expandableMoves": [0, 0, 1, 1, 0, 0, 0, 1, 0],
          "children": [
            {
              "value": 0,
              "actionTaken": 8,
              "visits": 1,
              "expandableMoves": [1, 0, 1, 1, 0, 0, 0, 1, 0],
              "children": [],
              "state": [[0, -1, 0], [0, 1, 1], [-1, 0, -1]]
            },
            {
              "value": 1,
              "actionTaken": 0,
              "visits": 1,
              "expandableMoves": [0, 0, 1, 1, 0, 0, 0, 1, 1],
              "children": [],
              "state": [[-1, -1, 0], [0, 1, 1], [-1, 0, 0]]
            }
          ],
          "state": [[0, 1, 0], [0, -1, -1], [1, 0, 0]]
        },
        {
          "value": 1,
          "actionTaken": 3,
          "visits": 2,
          "expandableMoves": [1, 0, 1, 0, 0, 1, 0, 1, 0],
          "children": [
            {
              "value": -1,
              "actionTaken": 8,
              "visits": 1,
              "expandableMoves": [1, 0, 1, 0, 0, 1, 0, 1, 0],
              "children": [],
              "state": [[0, -1, 0], [1, 1, 0], [-1, 0, -1]]
            }
          ],
          "state": [[0, 1, 0], [-1, -1, 0], [1, 0, 0]]
        },
        {
          "value": -2,
          "actionTaken": 0,
          "visits": 3,
          "expandableMoves": [0, 0, 1, 1, 0, 0, 0, 1, 0],
          "children": [
            {
              "value": 1,
              "actionTaken": 5,
              "visits": 1,
              "expandableMoves": [0, 0, 1, 1, 0, 0, 0, 1, 1],
              "children": [],
              "state": [[1, -1, 0], [0, 1, -1], [-1, 0, 0]]
            },
            {
              "value": 0,
              "actionTaken": 8,
              "visits": 1,
              "expandableMoves": [0, 0, 1, 1, 0, 1, 0, 1, 0],
              "children": [],
              "state": [[1, -1, 0], [0, 1, 0], [-1, 0, -1]]
            }
          ],
          "state": [[-1, 1, 0], [0, -1, 0], [1, 0, 0]]
        },
        {
          "value": 1,
          "actionTaken": 7,
          "visits": 2,
          "expandableMoves": [1, 0, 0, 1, 0, 1, 0, 0, 1],
          "children": [
            {
              "value": -1,
              "actionTaken": 2,
              "visits": 1,
              "expandableMoves": [1, 0, 0, 1, 0, 1, 0, 0, 1],
              "children": [],
              "state": [[0, -1, -1], [0, 1, 0], [-1, 1, 0]]
            }
          ],
          "state": [[0, 1, 0], [0, -1, 0], [1, -1, 0]]
        }
      ],
      "state": [[0, -1, 0], [0, 1, 0], [-1, 0, 0]]
    }
  ]
}
